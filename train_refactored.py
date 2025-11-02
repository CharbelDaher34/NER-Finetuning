"""
Refactored training script with modular architecture.

This script demonstrates how to use the modular training system.
To create a new task, simply:
1. Implement DataProcessor for your task
2. Implement Evaluator for your task
3. Configure ExperimentConfig
4. Run the orchestrator
"""

import os
from dotenv import load_dotenv

from config import ExperimentConfig
from tasks.ner_task import NERDataProcessor, NERTaskEvaluator
from trainer_orchestrator import TrainerOrchestrator


def main():
    """Main entry point for training."""
    
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = ExperimentConfig(
        random_seed=42,
        output_dir="results",
        
        # Set API keys from environment
        hf_token=os.getenv("HF_TOKEN"),
        wandb_api_key=os.getenv("WANDB_API_KEY"),
    )
    
    # You can override specific settings like this:
    config.model.base_model = "Qwen/Qwen3-0.6B"
    config.model.new_model = "Qwen/Qwen3-0.6B-finetuned"
    config.dataset.train_dataset_path = "./dataset.jsonl"
    config.dataset.test_dataset_path = "./test_dataset.jsonl"
    config.training.num_train_epochs = 4
    config.monitoring.wandb_project_name = "Fine-Tune Llama 3 8B on Crime Dataset"
    
    # Initialize data processor (tokenizer will be set by orchestrator)
    data_processor = NERDataProcessor(tokenizer=None)
    
    # Initialize evaluator (device will be set by orchestrator)
    evaluator = NERTaskEvaluator(
        config=config,
        tokenizer=None,
        device=None,
        data_processor=data_processor
    )
    
    # Create orchestrator
    orchestrator = TrainerOrchestrator(
        config=config,
        data_processor=data_processor,
        evaluator=evaluator
    )
    
    # Run full pipeline
    results_folder = orchestrator.run()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_folder}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

