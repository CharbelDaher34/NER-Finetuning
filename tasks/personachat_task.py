"""Personachat-specific implementations of data processor and evaluator."""

import re
import torch
from typing import Dict, Any, List, Tuple
import logging
from sentence_transformers import SentenceTransformer, util

from core.data_processor import DataProcessor
from core.evaluator import Evaluator


logger = logging.getLogger(__name__)


class PersonaChatDataProcessor(DataProcessor):
    """Data processor for persona-based chat task."""
    
    def parse_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse a persona chat example into training format.
        
        Args:
            example: Raw example with 'persona_b', 'dialogue', and 'reference' fields
            
        Returns:
            List containing single parsed example dict with messages
        """
        persona_b = example.get('persona_b', [])
        dialogue = example.get('dialogue', [])
        reference = example.get('reference', '')
        
        if not persona_b or not dialogue or not reference:
            return []
        
        # Parse dialogue into separate turns
        messages = []
        for turn in dialogue:
            if turn.startswith("Persona A: "):
                # User message
                messages.append({
                    "role": "user",
                    "content": turn.replace("Persona A: ", "", 1)
                })
            elif turn.startswith("Persona B: "):
                # Assistant message
                messages.append({
                    "role": "assistant",
                    "content": turn.replace("Persona B: ", "", 1)
                })
        
        return [{
            "persona_b": persona_b,
            "messages": messages,
            "reference": reference
        }]
    
    def format_for_training(self, parsed_data: Dict[str, Any]) -> str:
        """
        Format parsed data for training using HF chat template.
        
        Args:
            parsed_data: Parsed example dict with 'persona_b', 'messages', and 'reference'
            
        Returns:
            Formatted text using chat template with proper message structure
        """
        # Format persona_b list into system prompt
        persona_text = "\n".join([f"- {fact}" for fact in parsed_data['persona_b']])
        
        system_prompt = (
            "You are a conversational AI with the following persona:\n"
            f"{persona_text}\n\n"
            "Respond naturally and stay true to your persona."
        )
        
        # Build full conversation
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(parsed_data['messages'])
        messages.append({"role": "assistant", "content": parsed_data["reference"]})
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False
        )
    
    def format_for_inference(self, input_data: Dict[str, Any]) -> str:
        """
        Format data for inference using HF chat template.
        
        Args:
            input_data: Dict with 'persona_b' (list) and 'messages' keys
            
        Returns:
            Formatted prompt with generation prompt
        """
        # Format persona_b list into system prompt
        persona_text = "\n".join([f"- {fact}" for fact in input_data['persona_b']])
        
        system_prompt = (
            "You are a conversational AI with the following persona:\n"
            f"{persona_text}\n\n"
            "Respond naturally and stay true to your persona."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(input_data['messages'])
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )


class PersonaChatEvaluator(Evaluator):
    """Evaluator for persona-based chat task using semantic similarity."""
    
    def __init__(self, config, tokenizer, device, data_processor: PersonaChatDataProcessor):
        """
        Initialize PersonaChat evaluator.
        
        Args:
            config: Experiment configuration
            tokenizer: Tokenizer instance
            device: Device for inference
            data_processor: Data processor for formatting
        """
        super().__init__(config, tokenizer, device)
        self.data_processor = data_processor
        
        # Load sentence transformer for semantic similarity
        logger.info("Loading sentence transformer model for evaluation...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_model.to(device)
    
    def parse_dataset_example(self, example: Dict[str, Any]) -> Tuple[List[str], List[Tuple[List[Dict], str]]]:
        """
        Parse test dataset example.
        
        Args:
            example: Example with 'text' field (formatted conversation)
            
        Returns:
            Tuple of (persona_b_list, [(messages, reference)])
        """
        text = example.get("text", "")
        
        # Parse multi-turn conversation to extract persona and messages
        all_blocks = re.findall(
            r"<\|im_start\|>(system|user|assistant)\s*(.*?)<\|im_end\|>",
            text,
            re.DOTALL
        )
        
        persona_b = None
        messages = []
        
        for role, content in all_blocks:
            content = content.strip()
            
            if role == "system":
                # Extract persona from system message
                if "following persona:" in content:
                    parts = content.split("following persona:")
                    if len(parts) > 1:
                        persona_text = parts[1].split("Respond naturally")[0].strip()
                        # Parse back into list (each line starting with "- ")
                        persona_b = [line.strip("- ").strip() for line in persona_text.split("\n") if line.strip().startswith("-")]
            elif role in ["user", "assistant"] and persona_b is not None:
                # Collect conversation messages (except the last assistant message which is the reference)
                messages.append({"role": role, "content": content})
        
        if persona_b and len(messages) > 0:
            # Last assistant message is the reference
            reference = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
            conversation_messages = messages[:-1] if messages[-1]["role"] == "assistant" else messages
            
            return persona_b, [(conversation_messages, reference)]
        
        return [], []
    
    def generate_prediction(self, model, persona_b: List[str], messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Generate model prediction.
        
        Args:
            model: Model to use
            persona_b: List of persona facts
            messages: List of conversation messages
            
        Returns:
            Tuple of (raw_response, parsed_prediction) - both are the generated text for this task
        """
        # Format prompt
        prompt = self.data_processor.format_for_inference({
            "persona_b": persona_b,
            "messages": messages
        })
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_token_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.generation.max_new_tokens,
                min_new_tokens=self.config.generation.min_new_tokens,
                do_sample=self.config.generation.do_sample,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                repetition_penalty=self.config.generation.repetition_penalty,
                num_beams=self.config.generation.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # Decode
        new_tokens = outputs[0, input_token_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Return tuple: (raw_response, parsed_prediction)
        # For text generation, both are the same
        return generated_text, generated_text
    
    def calculate_metrics(self, predicted: str, ground_truth: str) -> Dict[str, Any]:
        """
        Calculate semantic similarity metrics.
        
        Args:
            predicted: Predicted response
            ground_truth: Ground truth response
            
        Returns:
            Metrics dictionary with tp/fp/fn (required by base Evaluator)
        """
        # Encode both texts
        pred_embedding = self.similarity_model.encode(predicted, convert_to_tensor=True)
        truth_embedding = self.similarity_model.encode(ground_truth, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity_score = util.cos_sim(pred_embedding, truth_embedding).item()
        
        # Also calculate length ratio as a supplementary metric
        len_pred = len(predicted.split())
        len_truth = len(ground_truth.split())
        length_ratio = min(len_pred, len_truth) / max(len_pred, len_truth) if max(len_pred, len_truth) > 0 else 0
        
        # Map similarity to tp/fp/fn (required by base Evaluator)
        # Use 0.7 threshold: if similarity >= 0.7, consider it a match (tp=1, fp=0, fn=0)
        # Otherwise, it's a mismatch (tp=0, fp=1, fn=1)
        similarity_threshold = 0.7
        if similarity_score >= similarity_threshold:
            tp, fp, fn = 1, 0, 0
        else:
            tp, fp, fn = 0, 1, 1
        
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "similarity": similarity_score,
            "length_ratio": length_ratio,
            "pred_length": len_pred,
            "truth_length": len_truth
        }
    
    def is_valid_prediction(self, prediction: Any) -> bool:
        """Check if prediction is valid (non-empty string)."""
        return isinstance(prediction, str) and len(prediction.strip()) > 0
    
    def is_schema_valid(self, prediction: Any) -> bool:
        """Check if prediction is a non-empty string."""
        return self.is_valid_prediction(prediction)
    
    def get_empty_prediction(self) -> str:
        """Return empty string as fallback."""
        return ""
    
    def check_exact_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted response closely matches ground truth.
        
        Args:
            predicted: Predicted response
            ground_truth: Ground truth response
            
        Returns:
            True if similarity score is very high (>0.95)
        """
        # Exact string match
        if predicted.strip().lower() == ground_truth.strip().lower():
            return True
        
        # High semantic similarity
        metrics = self.calculate_metrics(predicted, ground_truth)
        return metrics["similarity"] > 0.95
    
    def calculate_summary(self, all_results: List[Dict], total_predictions: int,
                         examples_processed: int, examples_skipped: int,
                         total_tp: int, total_fp: int, total_fn: int,
                         total_exact_matches: int, total_valid_predictions: int,
                         total_schema_valid: int) -> Dict:
        """
        Calculate PersonaChat-specific summary with similarity metrics.
        
        Returns:
            Summary dictionary with similarity-based metrics
        """
        # Calculate average similarity from all results
        similarities = []
        length_ratios = []
        
        for result in all_results:
            metrics = result.get('metrics', {})
            if 'similarity' in metrics:
                similarities.append(metrics['similarity'])
            if 'length_ratio' in metrics:
                length_ratios.append(metrics['length_ratio'])
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        avg_length_ratio = sum(length_ratios) / len(length_ratios) if length_ratios else 0.0
        
        # Count high similarity matches (>= 0.7)
        high_similarity_count = sum(1 for s in similarities if s >= 0.7)
        high_similarity_rate = high_similarity_count / total_predictions if total_predictions > 0 else 0.0
        
        exact_match_rate = total_exact_matches / total_predictions if total_predictions > 0 else 0.0
        valid_rate = total_valid_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "total_predictions": total_predictions,
            "examples_processed": examples_processed,
            "examples_skipped": examples_skipped,
            "avg_similarity": avg_similarity,
            "avg_length_ratio": avg_length_ratio,
            "high_similarity_rate": high_similarity_rate,
            "exact_match_rate": exact_match_rate,
            "valid_rate": valid_rate,
            "min_similarity": min(similarities) if similarities else 0.0,
            "max_similarity": max(similarities) if similarities else 0.0,
        }
    
    def log_summary(self, summary: Dict):
        """Log PersonaChat-specific evaluation summary."""
        logger.info("\n" + "="*80)
        logger.info("PERSONACHAT EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total predictions: {summary['total_predictions']}")
        logger.info(f"Examples processed: {summary['examples_processed']}")
        logger.info(f"Examples skipped: {summary['examples_skipped']}")
        logger.info(f"Average Similarity: {summary['avg_similarity']:.4f}")
        logger.info(f"Min Similarity: {summary['min_similarity']:.4f}")
        logger.info(f"Max Similarity: {summary['max_similarity']:.4f}")
        logger.info(f"High Similarity Rate (>= 0.7): {summary['high_similarity_rate']:.4f}")
        logger.info(f"Exact Match Rate: {summary['exact_match_rate']:.4f}")
        logger.info(f"Valid Rate: {summary['valid_rate']:.4f}")
        logger.info(f"Average Length Ratio: {summary['avg_length_ratio']:.4f}")
        logger.info("="*80 + "\n")

