"""Abstract base classes and implementations for data processing."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datasets import Dataset


class DataProcessor(ABC):
    """Abstract base class for data processing."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @abstractmethod
    def parse_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse a single example from the dataset.
        
        Args:
            example: Raw example from dataset
            
        Returns:
            List of processed examples (can expand single example to multiple)
        """
        pass
    
    @abstractmethod
    def format_for_training(self, parsed_data: Dict[str, Any]) -> str:
        """
        Format parsed data into training text.
        
        Args:
            parsed_data: Processed data from parse_example
            
        Returns:
            Formatted text for training
        """
        pass
    
    @abstractmethod
    def format_for_inference(self, input_data: Dict[str, Any]) -> str:
        """
        Format data for model inference.
        
        Args:
            input_data: Input data for inference
            
        Returns:
            Formatted prompt for inference
        """
        pass
    
    def process_batch(self, examples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Process a batch of examples for training.
        
        Args:
            examples: Batch of examples from dataset
            
        Returns:
            Batch of formatted texts
        """
        all_texts = []
        
        # Get the first key to iterate over (assumes all keys have same length)
        first_key = list(examples.keys())[0]
        batch_size = len(examples[first_key])
        
        for i in range(batch_size):
            # Extract single example from batch
            example = {key: examples[key][i] for key in examples.keys()}
            
            # Parse and format each example
            parsed_examples = self.parse_example(example)
            for parsed in parsed_examples:
                text = self.format_for_training(parsed)
                all_texts.append(text)
        
        return {"text": all_texts}
    
    def prepare_datasets(self, train_dataset: Dataset, eval_dataset: Dataset) -> tuple[Dataset, Dataset]:
        """
        Apply processing to train and eval datasets.
        
        Args:
            train_dataset: Raw training dataset
            eval_dataset: Raw evaluation dataset
            
        Returns:
            Processed training and evaluation datasets
        """
        train_processed = train_dataset.map(
            self.process_batch,
            batched=True,
            batch_size=1,
            remove_columns=train_dataset.column_names,
            desc="Processing training data"
        )#.select(range(10))
        
        eval_processed = eval_dataset.map(
            self.process_batch,
            batched=True,
            batch_size=1,
            remove_columns=eval_dataset.column_names,
            desc="Processing eval data"
        )#.select(range(2))
        
        return train_processed, eval_processed

