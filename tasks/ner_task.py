"""NER-specific implementations of data processor and evaluator."""

import re
import json
import torch
from typing import Dict, Any, List, Tuple
import logging

from core.data_processor import DataProcessor
from core.evaluator import Evaluator
from core.metrics import calculate_fuzzy_metrics, extract_json_from_text, normalize_dict_values_to_lists


logger = logging.getLogger(__name__)


class NERDataProcessor(DataProcessor):
    """Data processor for NER task with multi-turn conversations."""
    
    def parse_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse a multi-turn conversation into Q&A pairs.
        
        Args:
            example: Raw example with 'conversation' field
            
        Returns:
            List of Q&A pair dictionaries
        """
        conversation_text = example.get('conversation', '')
        lines = conversation_text.strip().split('\n')
        
        if not lines:
            return []
        
        system_prompt = lines[0]
        
        # Parse messages
        messages = []
        current_role = None
        current_content = []
        
        for line in lines[1:]:
            if line.startswith("User:"):
                if current_role == "assistant" and current_content:
                    messages.append({"role": "assistant", "content": "\n".join(current_content)})
                current_role = "user"
                current_content = [line.replace("User:", "", 1).strip()]
            elif line.startswith("Assistant:"):
                if current_role == "user" and current_content:
                    messages.append({"role": "user", "content": "\n".join(current_content)})
                current_role = "assistant"
                current_content = [line.replace("Assistant:", "", 1).strip()]
            else:
                current_content.append(line)
        
        if current_role and current_content:
            messages.append({"role": current_role, "content": "\n".join(current_content)})
        
        # Extract report and Q&A pairs
        if not messages or messages[0]["role"] != "user":
            return []
        
        # Extract report (remove "Text:" prefix if present)
        report = messages[0]["content"]
        if report.startswith("Text:"):
            report = report[5:].strip()
        
        qa_pairs = []
        i = 1
        if i < len(messages) and messages[i]["role"] == "assistant":
            i += 1  # Skip "I've read this text"
        
        while i < len(messages) - 1:
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                qa_pairs.append({
                    "system_prompt": system_prompt,
                    "report": report,
                    "question": messages[i]["content"],
                    "answer": messages[i+1]["content"]
                })
                i += 2
            else:
                i += 1
        
        return qa_pairs
    
    def format_for_training(self, parsed_data: Dict[str, Any]) -> str:
        """
        Format Q&A pair for training.
        
        Args:
            parsed_data: Parsed Q&A pair
            
        Returns:
            Formatted text using chat template
        """
        messages = [
            {"role": "system", "content": parsed_data["system_prompt"]},
            {"role": "user", "content": f"Text:\n{parsed_data['report']}"},
            {"role": "assistant", "content": "I've read this text."},
            {"role": "user", "content": parsed_data["question"]},
            {"role": "assistant", "content": parsed_data["answer"]}
        ]
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False)
    
    def format_for_inference(self, input_data: Dict[str, Any]) -> str:
        """
        Format data for inference.
        
        Args:
            input_data: Dict with 'report' and 'question' keys
            
        Returns:
            Formatted prompt
        """
        messages = [
            {"role": "system", "content": (
                "You answer ONLY with a JSON object. No prose, no backticks, nothing outside braces. "
                "Keys must match the question entity exactly. Values must be arrays of strings (or empty array)."
            )},
            {"role": "user", "content": f"Text:\n{input_data['report']}"},
            {"role": "assistant", "content": "I've read this text."},
            {"role": "user", "content": input_data['question']}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Seed opening brace to bias toward valid JSON
        return prompt + "\n{"


class NERTaskEvaluator(Evaluator):
    """Evaluator for NER task."""
    
    def __init__(self, config, tokenizer, device, data_processor: NERDataProcessor):
        """
        Initialize NER evaluator.
        
        Args:
            config: Experiment configuration
            tokenizer: Tokenizer instance
            device: Device for inference
            data_processor: Data processor for formatting
        """
        super().__init__(config, tokenizer, device)
        self.data_processor = data_processor
    
    def parse_dataset_example(self, example: Dict[str, Any]) -> Tuple[str, List[Tuple[str, Dict]]]:
        """
        Parse test dataset example.
        
        Args:
            example: Example with 'text' field (formatted conversation)
            
        Returns:
            Tuple of (document, qa_pairs)
        """
        text = example.get("text", "")
        
        # Parse multi-turn conversation
        all_blocks = re.findall(
            r"<\|im_start\|>(system|user|assistant)\s*(.*?)<\|im_end\|>",
            text,
            re.DOTALL
        )
        
        document = None
        qa_pairs = []
        current_question = None
        
        for role, content in all_blocks:
            content = content.strip()
            
            if role == "user" and document is None:
                # First user message is the document
                if content.startswith("Text:"):
                    document = content[5:].strip()
                else:
                    document = content
            elif role == "user" and document is not None:
                current_question = content
            elif role == "assistant" and current_question is not None:
                if content != "I've read this text.":
                    try:
                        ground_truth = json.loads(content)
                        qa_pairs.append((current_question, ground_truth))
                    except json.JSONDecodeError:
                        pass
                current_question = None
        
        return document, qa_pairs
    
    def generate_prediction(self, model, context: str, question: str) -> Tuple[str, Dict]:
        """
        Generate model prediction.
        
        Args:
            model: Model to use
            context: Document/report text
            question: Question to answer
            
        Returns:
            Tuple of (raw_response, parsed_json)
        """
        # Format prompt
        prompt = self.data_processor.format_for_inference({
            "report": context,
            "question": question
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
        
        # Reconstruct with seeded brace
        response_text = "{" + generated_text
        
        # Extract JSON
        json_obj = extract_json_from_text(response_text)
        
        if json_obj is None:
            json_obj = {}
        
        # Normalize to lists
        json_obj = normalize_dict_values_to_lists(json_obj)
        
        return response_text, json_obj
    
    def calculate_metrics(self, predicted: Dict, ground_truth: Dict) -> Dict[str, Any]:
        """
        Calculate metrics with fuzzy matching.
        
        Args:
            predicted: Predicted JSON
            ground_truth: Ground truth JSON
            
        Returns:
            Metrics dictionary
        """
        return calculate_fuzzy_metrics(
            predicted,
            ground_truth,
            threshold=self.config.evaluation.fuzzy_match_threshold
        )
    
    def is_valid_prediction(self, prediction: Any) -> bool:
        """Check if prediction is valid JSON dict."""
        return isinstance(prediction, dict)
    
    def is_schema_valid(self, prediction: Any) -> bool:
        """Check if all values are lists."""
        if not isinstance(prediction, dict):
            return False
        return all(isinstance(v, list) for v in prediction.values())
    
    def get_empty_prediction(self) -> Dict:
        """Return empty dict as fallback."""
        return {}
    
    def check_exact_match(self, predicted: Dict, ground_truth: Dict) -> bool:
        """
        Check exact or fuzzy complete match.
        
        Args:
            predicted: Predicted JSON
            ground_truth: Ground truth JSON
            
        Returns:
            True if exact match or fuzzy complete match
        """
        # Exact match
        if predicted == ground_truth:
            return True
        
        # Fuzzy complete match (all items matched with fuzzy, no FP/FN)
        metrics = self.calculate_metrics(predicted, ground_truth)
        return metrics["tp"] > 0 and metrics["fp"] == 0 and metrics["fn"] == 0

