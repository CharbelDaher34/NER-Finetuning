"""
Simple unified script to convert fine-tuned LoRA model to GGUF format.

Steps:
1. Merge LoRA adapter with base model
2. Convert to GGUF format (F16)
3. Optional: Quantize to smaller size

Usage:
    uv run convert_to_gguf.py [--quantize Q4_K_M]
"""

import os
import sys
import subprocess
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_results():
    """Find the most recent training results folder."""
    results_folders = sorted([f for f in os.listdir('results') if f.startswith('training_results_')])
    if not results_folders:
        logger.error("No training results folder found!")
        logger.info("Please run train.py first to create a trained model.")
        return None
    return os.path.join('results', results_folders[-1])


def merge_lora_adapter(base_model_name, adapter_path, output_merged_path):
    """Merge LoRA adapter with base model."""
    logger.info("="*80)
    logger.info("STEP 1: Merging LoRA adapter with base model")
    logger.info("="*80)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        logger.info(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        logger.info(f"Loading tokenizer from: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
        
        logger.info("Merging LoRA weights into base model...")
        merged_model = model_with_adapter.merge_and_unload()
        
        logger.info(f"Saving merged model to: {output_merged_path}")
        os.makedirs(output_merged_path, exist_ok=True)
        merged_model.save_pretrained(output_merged_path)
        tokenizer.save_pretrained(output_merged_path)
        
        logger.info("✓ Model merged successfully\n")
        return True
        
    except ImportError as e:
        logger.error(f"Missing required packages: {e}")
        logger.info("Install with: uv add transformers peft torch")
        return False
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        return False


def convert_to_gguf_f16(merged_model_path, gguf_output):
    """Convert merged model to GGUF F16 format."""
    logger.info("="*80)
    logger.info("STEP 2: Converting to GGUF (F16)")
    logger.info("="*80)
    
    # Clone llama.cpp if needed
    if not os.path.exists("llama.cpp"):
        logger.info("Cloning llama.cpp...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp"], check=True)
    
    # Install requirements
    logger.info("Installing Python requirements...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "llama.cpp/requirements.txt"],
        check=False
    )
    
    # Convert using Python script (no compilation needed!)
    convert_cmd = [
        sys.executable,
        "llama.cpp/convert_hf_to_gguf.py",
        merged_model_path,
        "--outfile", gguf_output,
        "--outtype", "f16"
    ]
    
    logger.info(f"Running: {' '.join(convert_cmd)}")
    result = subprocess.run(convert_cmd, check=False)
    
    if result.returncode == 0 and os.path.exists(gguf_output):
        size = os.path.getsize(gguf_output) / (1024**3)
        logger.info(f"✓ Conversion successful! Size: {size:.2f} GB\n")
        return True
    else:
        logger.error("✗ Conversion failed!")
        return False


def quantize_model(gguf_f16_path, quantization_type):
    """Quantize GGUF model to smaller size."""
    logger.info("="*80)
    logger.info(f"STEP 3: Quantizing to {quantization_type}")
    logger.info("="*80)
    
    # Check if quantize binary exists
    possible_paths = [
        "llama.cpp/build/bin/llama-quantize",
        "llama.cpp/build/bin/quantize",
        "llama.cpp/llama-quantize"
    ]
    
    quantize_bin = None
    for path in possible_paths:
        if os.path.exists(path):
            quantize_bin = path
            break
    
    if not quantize_bin:
        logger.warning("Quantize binary not found. Building llama.cpp...")
        try:
            os.makedirs("llama.cpp/build", exist_ok=True)
            subprocess.run(["cmake", "..", "-DGGML_CUDA=ON"], cwd="llama.cpp/build", check=True)
            subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd="llama.cpp/build", check=True)
            
            # Check again
            for path in possible_paths:
                if os.path.exists(path):
                    quantize_bin = path
                    break
        except Exception as e:
            logger.error(f"Failed to build llama.cpp: {e}")
            logger.info("Skipping quantization. You can use the F16 model or quantize manually later.")
            return None
    
    if not quantize_bin:
        logger.warning("Could not find or build quantize binary. Skipping quantization.")
        return None
    
    # Quantize
    output_path = gguf_f16_path.replace(".gguf", f"-{quantization_type}.gguf")
    quantize_cmd = [quantize_bin, gguf_f16_path, output_path, quantization_type]
    
    logger.info(f"Running: {' '.join(quantize_cmd)}")
    result = subprocess.run(quantize_cmd, check=False)
    
    if result.returncode == 0 and os.path.exists(output_path):
        size = os.path.getsize(output_path) / (1024**3)
        logger.info(f"✓ Quantization successful! Size: {size:.2f} GB\n")
        return output_path
    else:
        logger.warning("Quantization failed, but F16 model is still available.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert fine-tuned LoRA model to GGUF format")
    parser.add_argument("--quantize", type=str, default=None,
                       help="Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.). If not specified, only F16 conversion is done.")
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("GGUF CONVERSION PIPELINE")
    logger.info("="*80 + "\n")
    
    # Find latest results
    latest_results = find_latest_results()
    if not latest_results:
        return 1
    latest_results = "./best_model"
    
    logger.info(f"Using training results: {latest_results}\n")
    
    # Paths
    base_model_name = "Qwen/Qwen3-0.6B"
    adapter_path = os.path.join(latest_results, "lora_adapter")
    merged_model_path = os.path.join(latest_results, "merged_model")
    gguf_f16_path = os.path.join(latest_results, "model.gguf")
    
    # Check adapter exists
    if not os.path.exists(adapter_path):
        logger.error(f"LoRA adapter not found at: {adapter_path}")
        return 1
    
    # Step 1: Merge LoRA adapter (if not already done)
    if os.path.exists(merged_model_path):
        logger.info("Merged model already exists, skipping merge step.\n")
    else:
        if not merge_lora_adapter(base_model_name, adapter_path, merged_model_path):
            return 1
    
    # Step 2: Convert to GGUF F16
    if os.path.exists(gguf_f16_path):
        logger.info("GGUF F16 model already exists, skipping conversion.\n")
    else:
        if not convert_to_gguf_f16(merged_model_path, gguf_f16_path):
            return 1
    
    # Step 3: Quantize (optional)
    quantized_path = None
    if args.quantize:
        quantized_path = quantize_model(gguf_f16_path, args.quantize)
    
    # Summary
    logger.info("="*80)
    logger.info("✅ CONVERSION COMPLETE!")
    logger.info("="*80)
    logger.info("\nOutput files:")
    logger.info(f"  Merged model: {merged_model_path}")
    logger.info(f"  GGUF F16:     {gguf_f16_path}")
    if quantized_path:
        logger.info(f"  Quantized:    {quantized_path}")
    
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("\n1. Test the model:")
    logger.info(f"   uv run inference_gguf.py --model {gguf_f16_path} --test")
    
    if args.quantize and not quantized_path:
        logger.info("\n2. To quantize manually:")
        logger.info("   cd llama.cpp && mkdir -p build && cd build")
        logger.info("   cmake .. -DGGML_CUDA=ON && cmake --build . --config Release")
        logger.info(f"   ./bin/llama-quantize ../../{gguf_f16_path} ../../{latest_results}/model-{args.quantize}.gguf {args.quantize}")
    
    logger.info("\n3. Install llama-cpp-python for inference:")
    logger.info("   uv add llama-cpp-python")
    logger.info("   # For GPU: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python --force-reinstall")
    logger.info("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

