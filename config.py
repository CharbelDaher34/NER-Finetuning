"""Configuration models using Pydantic for type safety and validation."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import torch


class ModelConfig(BaseModel):
    """Model-related configuration."""
    base_model: str = Field(default="Qwen/Qwen3-0.6B", description="Base model identifier")
    new_model: str = Field(default="Qwen/Qwen3-0.6B-finetuned", description="Output model name")
    torch_dtype: str = Field(default="bfloat16", description="Torch dtype (bfloat16, float16, float32)")
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = Field(
        default="sdpa", 
        description="Attention implementation"
    )
    
    @property
    def get_torch_dtype(self):
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


class DatasetConfig(BaseModel):
    """Dataset-related configuration."""
    train_dataset_path: str = Field(default="./dataset.jsonl", description="Path to training dataset")
    test_dataset_path: str = Field(default="./test_dataset.jsonl", description="Path to test dataset")
    max_seq_length: int = Field(default=1536, description="Maximum sequence length")
    train_test_split: float = Field(default=0.1, ge=0.0, le=1.0, description="Validation split ratio")
    keep_in_memory: bool = Field(default=False, description="Whether to keep dataset in memory")


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""
    r: int = Field(default=16, ge=1, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha (scaling factor)")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        description="Target modules for LoRA"
    )
    bias: str = Field(default="none", description="Bias configuration")
    task_type: str = Field(default="CAUSAL_LM", description="Task type")


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    num_train_epochs: int = Field(default=4, ge=1, description="Number of training epochs")
    learning_rate: float = Field(default=5e-5, gt=0.0, description="Learning rate")
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0, description="Warmup ratio")
    lr_scheduler_type: str = Field(default="cosine", description="Learning rate scheduler")
    per_device_train_batch_size: int = Field(default=2, ge=1, description="Training batch size per device")
    per_device_eval_batch_size: int = Field(default=2, ge=1, description="Eval batch size per device")
    gradient_accumulation_steps: int = Field(default=8, ge=1, description="Gradient accumulation steps")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay")
    max_grad_norm: float = Field(default=0.3, gt=0.0, description="Max gradient norm for clipping")
    optimizer: str = Field(default="adamw_8bit", description="Optimizer type")
    fp16: bool = Field(default=False, description="Use FP16 training")
    bf16: bool = Field(default=True, description="Use BF16 training")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    use_reentrant: bool = Field(default=False, description="Use reentrant checkpointing")
    group_by_length: bool = Field(default=True, description="Group sequences by length")


class MonitoringConfig(BaseModel):
    """Training monitoring configuration."""
    logging_steps: int = Field(default=25, ge=1, description="Log every N steps")
    eval_steps: int = Field(default=125, ge=1, description="Evaluate every N steps")
    save_steps: int = Field(default=125, ge=1, description="Save checkpoint every N steps")
    save_total_limit: int = Field(default=2, ge=1, description="Max number of checkpoints to keep")
    early_stopping_patience: int = Field(default=3, ge=1, description="Early stopping patience")
    early_stopping_threshold: float = Field(default=0.001, ge=0.0, description="Early stopping threshold")
    wandb_project_name: str = Field(default="Fine-Tune LLM", description="W&B project name")
    report_to: str = Field(default="wandb", description="Reporting destination")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    fuzzy_match_threshold: float = Field(
        default=85.0, 
        ge=0.0, 
        le=100.0, 
        description="Fuzzy matching threshold (0-100)"
    )
    run_pre_training_eval: bool = Field(default=True, description="Run evaluation before training")
    run_post_training_eval: bool = Field(default=True, description="Run evaluation after training")


class GenerationConfig(BaseModel):
    """Text generation configuration."""
    max_new_tokens: int = Field(default=256, ge=1, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.1, gt=0.0, description="Generation temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Use sampling for generation")
    num_beams: int = Field(default=1, ge=1, description="Number of beams for beam search")
    min_new_tokens: int = Field(default=1, ge=1, description="Minimum new tokens to generate")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""
    task_name: str = Field(default="ner", description="Name of the task (e.g., ner, qa)")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    output_dir: str = Field(default="results", description="Output directory for results")
    
    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    
    # Environment
    hf_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    wandb_api_key: Optional[str] = Field(default=None, description="W&B API key")
    
    model_config = {"arbitrary_types_allowed": True}
    
    @field_validator("random_seed")
    @classmethod
    def validate_seed(cls, v):
        """Ensure seed is non-negative."""
        if v < 0:
            raise ValueError("Random seed must be non-negative")
        return v
    
    def set_random_seeds(self):
        """Set all random seeds for reproducibility."""
        import random
        import numpy as np
        import torch
        
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

