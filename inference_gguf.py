"""
Simple GGUF model inference for NER tasks.

Usage:
    from inference_gguf import infer
    
    response_text, json_response = infer(
        model_path="path/to/model.gguf",
        system_prompt="A virtual assistant...",
        report_text="Crime report text here",
        question="What describes Location in the text?"
    )
    print(json_response)
"""
import json
import re


def infer(model_path, system_prompt, report_text, question, max_tokens=512, temperature=0.0, n_gpu_layers=-1):
    """
    Run inference using GGUF model with formatted chat template.
    
    Args:
        model_path: Path to GGUF model file
        system_prompt: System message (e.g., task description)
        report_text: The crime report or document text
        question: The user's question about the text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        n_gpu_layers: GPU layers to use (-1 = all, 0 = CPU only)
    
    Returns:
        Tuple of (response_text, json_dict):
        - response_text: Raw model output
        - json_dict: Parsed JSON response as dictionary
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed.\n"
            "Install with: uv add llama-cpp-python\n"
            "For GPU: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python --force-reinstall"
        )
    
    # Format prompt using Qwen3 chat template
    # Matches the format from train.py: system -> user(report) -> assistant(ack) -> user(question) -> assistant(answer)
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Text:
{report_text}<|im_end|>
<|im_start|>assistant
I've read this text.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    # Load model with full context capacity
    llm = Llama(
        model_path=model_path,
        n_ctx=40960,  
        n_threads=8,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    
    # Generate response
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|im_end|>"],
        echo=False,
    )
    
    response_text = output['choices'][0]['text'].strip()
    
    # Remove thinking tags - Qwen3 models generate internal reasoning in <think> tags
    # We want only the final JSON output
    if '<think>' in response_text:
        # Remove everything between and including <think> and </think>
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    
    # Clean up any stray tags
    response_text = response_text.replace('<think>', '').replace('</think>', '').strip()
    
    # Parse JSON from response (matches train.py logic)
    try:
        # Extract JSON from response - find the first { and last }
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            return response_text, {}
        
        json_str = response_text[start_idx:end_idx+1]
        json_text = json.loads(json_str)
        
        # Normalize values to lists (convert strings to single-item lists)
        # Note: Don't split on commas as they may be part of addresses or values
        for key, value in json_text.items():
            if isinstance(value, list):
                # Already a list, keep as is
                pass
            elif isinstance(value, str):
                # Convert string to single-item list
                json_text[key] = [value]
            elif value is not None:
                # Convert other types to single-item list
                json_text[key] = [value]
            else:
                # None or missing -> empty list
                json_text[key] = []
        
        return response_text, json_text
    except (json.JSONDecodeError, IndexError, KeyError, AttributeError, ValueError) as e:
        # If JSON parsing fails, return empty dict
        return response_text, {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GGUF Model Inference for NER")
    parser.add_argument("--model", required=False, help="Path to GGUF model file", 
                        default="results/training_results_20251024_071121/model.gguf")
    parser.add_argument("--system-prompt", required=False, 
                        default="A virtual assistant answers questions from a user based on the provided text, answer with a json object, key being the entity asked for by user and the value extracted from the text.",
                        help="System prompt")
    parser.add_argument("--report", required=True, help="Crime report text")
    parser.add_argument("--question", required=True, help="Question about the text")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature")
    
    args = parser.parse_args()
    
    response_text, json_response = infer(
        model_path=args.model,
        system_prompt=args.system_prompt,
        report_text=args.report,
        question=args.question,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print("\n" + "="*80)
    print("RAW RESPONSE:")
    print("="*80)
    print(response_text)
    print("\n" + "="*80)
    print("JSON RESPONSE:")
    print("="*80)
    print(json.dumps(json_response, indent=2))
    print("="*80)
