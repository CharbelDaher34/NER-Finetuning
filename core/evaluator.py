"""Abstract base classes and implementations for model evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import logging
from tqdm import tqdm
import torch
import gc


logger = logging.getLogger(__name__)


class Evaluator(ABC):
    """Abstract base class for model evaluation."""
    
    def __init__(self, config, tokenizer, device):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
    
    @abstractmethod
    def parse_dataset_example(self, example: Dict[str, Any]) -> Tuple[Any, List[Tuple[Any, Any]]]:
        """
        Parse a single dataset example for evaluation.
        
        Args:
            example: Single example from test dataset
            
        Returns:
            Tuple of (context, qa_pairs) where qa_pairs is list of (question, ground_truth)
        """
        pass
    
    @abstractmethod
    def generate_prediction(self, model, context: Any, question: Any) -> Tuple[str, Any]:
        """
        Generate model prediction for a question given context.
        
        Args:
            model: The model to evaluate
            context: Context/document for the question
            question: Question to answer
            
        Returns:
            Tuple of (raw_response_text, parsed_prediction)
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, predicted: Any, ground_truth: Any) -> Dict[str, Any]:
        """
        Calculate metrics for a single prediction.
        
        Args:
            predicted: Model prediction
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with metrics (must include 'tp', 'fp', 'fn')
        """
        pass
    
    @abstractmethod
    def is_valid_prediction(self, prediction: Any) -> bool:
        """
        Check if prediction is valid format.
        
        Args:
            prediction: Model prediction
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def is_schema_valid(self, prediction: Any) -> bool:
        """
        Check if prediction follows expected schema.
        
        Args:
            prediction: Model prediction
            
        Returns:
            True if schema valid, False otherwise
        """
        pass
    
    def check_exact_match(self, predicted: Any, ground_truth: Any) -> bool:
        """
        Check if prediction exactly matches ground truth.
        
        Args:
            predicted: Model prediction
            ground_truth: Ground truth answer
            
        Returns:
            True if exact match, False otherwise
        """
        return predicted == ground_truth
    
    def evaluate(self, model, test_dataset, log_interval: int = 50) -> Tuple[List[Dict], Dict]:
        """
        Evaluate model on test dataset.
        
        Args:
            model: Model to evaluate
            test_dataset: Test dataset
            log_interval: Log progress every N predictions
            
        Returns:
            Tuple of (detailed_results, summary_metrics)
        """
        logger.info("Starting model evaluation")
        logger.info(f"Dataset size: {len(test_dataset)} examples")
        
        # Set model to eval mode
        model.eval()
        torch.set_grad_enabled(False)
        
        all_results = []
        
        # Initialize counters
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_exact_matches = 0
        total_valid_predictions = 0
        total_schema_valid = 0
        total_predictions = 0
        
        examples_processed = 0
        examples_skipped = 0
        
        for ex_idx, example in enumerate(tqdm(test_dataset, desc="Evaluating model")):
            logger.debug(f"Processing example {ex_idx}")
            
            context, qa_pairs = self.parse_dataset_example(example)
            
            if not context or not qa_pairs:
                logger.warning(f"Skipping example {ex_idx} (missing context or Q&A pairs)")
                examples_skipped += 1
                continue
            
            examples_processed += 1
            logger.info(f"Evaluating example {ex_idx}: {len(qa_pairs)} Q&A pairs")
            
            for qa_idx, (question, ground_truth) in enumerate(qa_pairs):
                total_predictions += 1
                
                logger.info(f"\n{'='*80}")
                logger.info(f"Example {ex_idx}, Q&A pair {qa_idx}")
                logger.info(f"{'='*80}")
                
                predicted = None
                response_text = ""
                is_valid = False
                is_schema_ok = False
                
                try:
                    response_text, predicted = self.generate_prediction(model, context, question)
                    is_valid = self.is_valid_prediction(predicted)
                    is_schema_ok = self.is_schema_valid(predicted)
                    logger.info("✓ Successfully generated prediction")
                except Exception as e:
                    logger.error(f"✗ Error generating prediction: {str(e)}")
                    predicted = self.get_empty_prediction()
                    is_valid = False
                    is_schema_ok = False
                
                if is_valid:
                    total_valid_predictions += 1
                if is_schema_ok:
                    total_schema_valid += 1
                
                # Calculate metrics
                metrics = self.calculate_metrics(predicted, ground_truth)
                total_tp += metrics["tp"]
                total_fp += metrics["fp"]
                total_fn += metrics["fn"]
                
                # Check exact match
                is_exact = self.check_exact_match(predicted, ground_truth)
                if is_exact:
                    total_exact_matches += 1
                    logger.info("✓ EXACT MATCH")
                else:
                    logger.info("✗ NO EXACT MATCH")
                
                logger.info(f"Metrics: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
                logger.info(f"{'='*80}\n")
                
                all_results.append({
                    "example_idx": ex_idx,
                    "qa_idx": qa_idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "response_text": response_text,
                    "exact_match": is_exact,
                    "valid": is_valid,
                    "schema_valid": is_schema_ok,
                    "metrics": metrics
                })
                
                # Log progress
                if total_predictions % log_interval == 0:
                    current_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                    current_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                    current_exact = total_exact_matches / total_predictions if total_predictions > 0 else 0.0
                    logger.info(
                        f"PROGRESS: {total_predictions} predictions. "
                        f"P={current_precision:.3f}, R={current_recall:.3f}, Exact={current_exact:.3f}"
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Calculate final metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        exact_match_rate = total_exact_matches / total_predictions if total_predictions > 0 else 0.0
        valid_rate = total_valid_predictions / total_predictions if total_predictions > 0 else 0.0
        schema_valid_rate = total_schema_valid / total_predictions if total_predictions > 0 else 0.0
        
        summary = {
            "total_predictions": total_predictions,
            "examples_processed": examples_processed,
            "examples_skipped": examples_skipped,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match_rate": exact_match_rate,
            "valid_rate": valid_rate,
            "schema_valid_rate": schema_valid_rate,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_exact_matches": total_exact_matches,
        }
        
        self.log_summary(summary)
        
        return all_results, summary
    
    def log_summary(self, summary: Dict):
        """Log evaluation summary."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total predictions: {summary['total_predictions']}")
        logger.info(f"Examples processed: {summary['examples_processed']}")
        logger.info(f"Examples skipped: {summary['examples_skipped']}")
        logger.info(f"Precision: {summary['precision']:.4f}")
        logger.info(f"Recall: {summary['recall']:.4f}")
        logger.info(f"F1: {summary['f1']:.4f}")
        logger.info(f"Exact Match Rate: {summary['exact_match_rate']:.4f}")
        logger.info(f"Valid Rate: {summary['valid_rate']:.4f}")
        logger.info(f"Schema Valid Rate: {summary['schema_valid_rate']:.4f}")
        logger.info("="*80 + "\n")
    
    @abstractmethod
    def get_empty_prediction(self) -> Any:
        """Return an empty prediction (fallback when generation fails)."""
        pass

