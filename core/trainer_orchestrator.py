"""Main training orchestrator that coordinates all components."""

import os
import json
import logging
import shutil
from typing import Optional, Dict, Any
import torch
import wandb
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from peft import LoraConfig as PeftLoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

from core.config import ExperimentConfig
from core.data_processor import DataProcessor
from core.evaluator import Evaluator


logger = logging.getLogger(__name__)


class TrainerOrchestrator:
    """
    Orchestrates the entire training pipeline.
    
    This class coordinates:
    - Environment setup
    - Data loading and processing
    - Model loading
    - Training
    - Evaluation
    - Results saving
    """
    
    def __init__(
        self, 
        config: ExperimentConfig,
        data_processor: DataProcessor,
        evaluator: Evaluator
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Experiment configuration
            data_processor: Data processor instance
            evaluator: Evaluator instance
        """
        self.config = config
        self.data_processor = data_processor
        self.evaluator = evaluator
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.trainer = None
        
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
    
    def setup_environment(self):
        """Setup logging, authentication, and random seeds."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        
        # Set random seeds
        self.config.set_random_seeds()
        logger.info(f"Random seeds set to {self.config.random_seed}")
        
        # Login to services
        if self.config.hf_token:
            login(token=self.config.hf_token)
            logger.info("Logged in to HuggingFace")
        
        if self.config.wandb_api_key:
            wandb.login(key=self.config.wandb_api_key)
            self.wandb_run = wandb.init(
                project=self.config.monitoring.wandb_project_name,
                job_type="training",
                config=self.config.model_dump(),
                anonymous="allow"
            )
            logger.info("Initialized W&B")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model.base_model}")
        logger.info(f"Attention implementation: {self.config.model.attn_implementation}")
        
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.base_model,
                device_map={"": device_id},
                attn_implementation=self.config.model.attn_implementation,
                torch_dtype=self.config.model.get_torch_dtype,
            )
            logger.info(f"✓ Model loaded with {self.config.model.attn_implementation}")
        except Exception as e:
            logger.warning(f"Failed with {self.config.model.attn_implementation}: {str(e)}")
            logger.info("Falling back to 'eager' attention")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.base_model,
                device_map={"": device_id},
                attn_implementation="eager",
                torch_dtype=self.config.model.get_torch_dtype,
            )
            logger.info("✓ Model loaded with eager attention")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Sync model config with tokenizer
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        logger.info(f"Tokenizer configured: pad_token_id={self.tokenizer.pad_token_id}")
    
    def _format_test_conversation(self, row):
        """Format the full multi-turn conversation for evaluation."""
        if not hasattr(self.data_processor, 'parse_example'):
            row['text'] = row.get('conversation', '')
            return row
            
        qa_pairs = self.data_processor.parse_example(row)
        if qa_pairs:
            # Create the full multi-turn conversation with all Q&A pairs
            messages = [
                {"role": "system", "content": qa_pairs[0]["system_prompt"]},
                {"role": "user", "content": f"Text:\n{qa_pairs[0]['report']}"},
                {"role": "assistant", "content": "I've read this text."}
            ]
            
            # Add all Q&A pairs
            for qa in qa_pairs:
                messages.append({"role": "user", "content": qa["question"]})
                messages.append({"role": "assistant", "content": qa["answer"]})
            
            row['text'] = self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            row['text'] = ""
        return row
    
    def load_datasets(self):
        """Load and split datasets."""
        # Set tokenizer on data processor before processing datasets
        if hasattr(self.data_processor, 'tokenizer') and self.data_processor.tokenizer is None:
            self.data_processor.tokenizer = self.tokenizer
            logger.info("Set tokenizer on data processor")
        
        logger.info(f"Loading training dataset from {self.config.dataset.train_dataset_path}")
        dataset = load_dataset(
            'json',
            data_files=self.config.dataset.train_dataset_path,
            keep_in_memory=self.config.dataset.keep_in_memory
        )
        logger.info(f"Loaded {len(dataset['train'])} training examples")
        
        # Split into train/eval
        split_ratio = self.config.dataset.train_test_split
        logger.info(f"Splitting dataset: {int((1-split_ratio)*100)}% train / {int(split_ratio*100)}% eval")
        splits = dataset['train'].train_test_split(test_size=split_ratio, seed=self.config.random_seed)
        train_raw = splits["train"]
        eval_raw = splits["test"]
        
        # Process datasets
        self.train_dataset, self.eval_dataset = self.data_processor.prepare_datasets(train_raw, eval_raw)
        logger.info(f"Processed {len(self.train_dataset)} train examples, {len(self.eval_dataset)} eval examples")
        
        # Load test dataset
        logger.info(f"Loading test dataset from {self.config.dataset.test_dataset_path}")
        test_data = load_dataset(
            'json',
            data_files=self.config.dataset.test_dataset_path,
            keep_in_memory=self.config.dataset.keep_in_memory
        )
        
        # Process test dataset with full conversation format
        logger.info("Processing test dataset: formatting for evaluation (keeping multi-turn structure)")
        self.test_dataset = test_data['train'].map(
            lambda row: self._format_test_conversation(row)
        )#.select(range(2))
        logger.info(f"Loaded {len(self.test_dataset)} test examples")
    
    def setup_trainer(self):
        """Setup LoRA and trainer."""
        logger.info("Setting up LoRA configuration")
        peft_config = PeftLoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type,
            target_modules=self.config.lora.target_modules
        )
        
        logger.info("Setting up training configuration")
        sft_config = SFTConfig(
            output_dir=self.config.model.new_model,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            num_train_epochs=self.config.training.num_train_epochs,
            learning_rate=self.config.training.learning_rate,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            max_grad_norm=self.config.training.max_grad_norm,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": self.config.training.use_reentrant},
            optim=self.config.training.optimizer,
            weight_decay=self.config.training.weight_decay,
            group_by_length=self.config.training.group_by_length,
            logging_steps=self.config.monitoring.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.monitoring.eval_steps,
            save_strategy="steps",
            save_steps=self.config.monitoring.save_steps,
            save_total_limit=self.config.monitoring.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataset_text_field="text",
            max_length=self.config.dataset.max_seq_length,
            packing=False,
            report_to=self.config.monitoring.report_to,
            dataloader_num_workers=0,
        )
        
        logger.info("Initializing SFTTrainer")
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.monitoring.early_stopping_patience,
            early_stopping_threshold=self.config.monitoring.early_stopping_threshold
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=peft_config,
            args=sft_config,
            callbacks=[early_stopping],
        )
        
        # Update model reference to LoRA-wrapped model
        self.model = self.trainer.model
        logger.info("Trainer initialized successfully")
    
    def run_evaluation(self, stage: str = "pre") -> tuple[list, dict]:
        """
        Run evaluation on test dataset.
        
        Args:
            stage: Evaluation stage ("pre" or "post")
            
        Returns:
            Tuple of (detailed_results, summary)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"{stage.upper()}-TRAINING EVALUATION")
        logger.info(f"{'='*80}\n")
        
        self.model.config.use_cache = True
        results, summary = self.evaluator.evaluate(self.model, self.test_dataset)
        
        # Log to wandb
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            wandb_metrics = {f"{stage}_training/{k}": v for k, v in summary.items() 
                           if isinstance(v, (int, float))}
            wandb.log(wandb_metrics)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        return results, summary
    
    def train(self):
        """Run training."""
        logger.info("Starting training...")
        torch.set_grad_enabled(True)
        self.model.train()
        self.model.config.use_cache = False
        
        self.trainer.train()
        
        logger.info("Training completed")
        gc.collect()
        torch.cuda.empty_cache()
    
    def save_results(self, results_pre: Optional[list], summary_pre: Optional[dict],
                    results_post: Optional[list], summary_post: Optional[dict]):
        """Save all results to disk."""
        # Determine training number
        task_dir = os.path.join(self.config.output_dir, self.config.task_name)
        last_number = 0
        try:
            if os.path.exists(task_dir):
                for folder in os.listdir(task_dir):
                    if folder.startswith("training_results_"):
                        last_number = max(last_number, int(folder.split("_")[-1]))
            training_number = last_number + 1
        except (FileNotFoundError, ValueError, IndexError):
            training_number = 1
        
        results_folder = os.path.join(task_dir, f"training_results_{training_number}")
        os.makedirs(results_folder, exist_ok=True)
        logger.info(f"Saving results to: {results_folder}")
        
        # Save metrics
        if summary_pre:
            with open(os.path.join(results_folder, "metrics_pre_training.json"), 'w') as f:
                json.dump(summary_pre, f, indent=2)
            if results_pre:
                with open(os.path.join(results_folder, "detailed_results_pre_training.json"), 'w') as f:
                    json.dump(results_pre, f, indent=2, default=str)
        
        if summary_post:
            with open(os.path.join(results_folder, "metrics_post_training.json"), 'w') as f:
                json.dump(summary_post, f, indent=2)
            if results_post:
                with open(os.path.join(results_folder, "detailed_results_post_training.json"), 'w') as f:
                    json.dump(results_post, f, indent=2, default=str)
        
        # Calculate and save improvement
        if summary_pre and summary_post:
            improvement = {
                "precision_delta": summary_post['precision'] - summary_pre['precision'],
                "recall_delta": summary_post['recall'] - summary_pre['recall'],
                "f1_delta": summary_post['f1'] - summary_pre['f1'],
            }
            with open(os.path.join(results_folder, "metrics_improvement.json"), 'w') as f:
                json.dump(improvement, f, indent=2)
        
        # Save training configuration
        with open(os.path.join(results_folder, "training_config.json"), 'w') as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)
        
        # Save LoRA adapter
        adapter_folder = os.path.join(results_folder, "lora_adapter")
        self.trainer.model.save_pretrained(adapter_folder)
        self.tokenizer.save_pretrained(adapter_folder)
        logger.info(f"LoRA adapter saved to: {adapter_folder}")
        
        # Copy logs
        if os.path.exists("training.log"):
            shutil.copy("training.log", os.path.join(results_folder, "training.log"))
        
        logger.info(f"All results saved to: {results_folder}")
        return results_folder
    
    def run(self):
        """Execute the full training pipeline."""
        try:
            # Setup
            self.setup_environment()
            self.load_model_and_tokenizer()
            
            # Set tokenizer and device on evaluator (data processor set in load_datasets)
            if hasattr(self.evaluator, 'tokenizer') and self.evaluator.tokenizer is None:
                self.evaluator.tokenizer = self.tokenizer
            if hasattr(self.evaluator, 'device') and self.evaluator.device is None:
                self.evaluator.device = self.device
            
            self.load_datasets()
            self.setup_trainer()
            
            # Pre-training evaluation
            results_pre, summary_pre = None, None
            if self.config.evaluation.run_pre_training_eval:
                results_pre, summary_pre = self.run_evaluation("pre")
            
            # Training
            self.train()
            
            # Post-training evaluation
            results_post, summary_post = None, None
            if self.config.evaluation.run_post_training_eval:
                results_post, summary_post = self.run_evaluation("post")
            
            # Save everything
            results_folder = self.save_results(results_pre, summary_pre, results_post, summary_post)
            
            # Check if this is the best model and promote it
            if summary_post:
                self.check_and_promote_best_model(results_folder, summary_post)
            
            # Cleanup
            if hasattr(self, 'wandb_run') and self.wandb_run is not None:
                wandb.finish()
            
            logger.info("Training pipeline completed successfully!")
            return results_folder
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
            if hasattr(self, 'wandb_run') and self.wandb_run is not None:
                wandb.finish(exit_code=1)
            raise

    def check_and_promote_best_model(self, current_results_folder: str, current_metrics: dict):
        """
        Check if current model is better than existing best model, and if so, promote it.
        Also triggers GGUF conversion for the best model.
        """
        best_model_dir = os.path.join("best_model", self.config.task_name)
        best_metrics_path = os.path.join(best_model_dir, "metrics.json")
        
        is_better = False
        
        if not os.path.exists(best_metrics_path):
            logger.info("No existing best model found. Promoting current model to best.")
            is_better = True
        else:
            try:
                with open(best_metrics_path, 'r') as f:
                    best_metrics = json.load(f)
                
                # Compare F1 score (or fallback to other metrics)
                current_f1 = current_metrics.get('f1', 0.0)
                best_f1 = best_metrics.get('f1', 0.0)
                
                if current_f1 > best_f1:
                    logger.info(f"Current model (F1={current_f1:.4f}) is better than previous best (F1={best_f1:.4f}).")
                    is_better = True
                else:
                    logger.info(f"Current model (F1={current_f1:.4f}) is NOT better than previous best (F1={best_f1:.4f}).")
            except Exception as e:
                logger.warning(f"Could not read best model metrics: {e}. Promoting current model.")
                is_better = True
        
        if is_better:
            self.promote_to_best_model(current_results_folder, best_model_dir, current_metrics)
            self.run_gguf_conversion(best_model_dir)

    def promote_to_best_model(self, source_folder: str, target_folder: str, metrics: dict):
        """Copy model files to best_model directory."""
        logger.info(f"Promoting model to: {target_folder}")
        
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
        os.makedirs(target_folder, exist_ok=True)
        
        # Copy LoRA adapter
        src_adapter = os.path.join(source_folder, "lora_adapter")
        dst_adapter = os.path.join(target_folder, "lora_adapter")
        if os.path.exists(src_adapter):
            shutil.copytree(src_adapter, dst_adapter)
            
        # Save metrics
        with open(os.path.join(target_folder, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info("✓ Model promoted successfully")

    def run_gguf_conversion(self, model_folder: str):
        """Run GGUF conversion on the promoted model."""
        logger.info("Starting GGUF conversion for best model...")
        
        try:
            # Import here to avoid circular imports or early failures
            from scripts.convert_to_gguf import merge_lora_adapter, convert_to_gguf_f16
            
            adapter_path = os.path.join(model_folder, "lora_adapter")
            merged_path = os.path.join(model_folder, "merged_model")
            gguf_path = os.path.join(model_folder, "model.gguf")
            
            # 1. Merge
            if not os.path.exists(merged_path):
                success = merge_lora_adapter(
                    self.config.model.base_model,
                    adapter_path,
                    merged_path
                )
                if not success:
                    logger.error("Failed to merge model during auto-conversion")
                    return
            
            # 2. Convert
            if not os.path.exists(gguf_path):
                success = convert_to_gguf_f16(merged_path, gguf_path)
                if success:
                    logger.info(f"✓ Auto-conversion complete: {gguf_path}")
                else:
                    logger.error("Failed to convert model to GGUF")
            else:
                logger.info("GGUF model already exists")
                
        except Exception as e:
            logger.error(f"Error during auto-conversion: {e}")

