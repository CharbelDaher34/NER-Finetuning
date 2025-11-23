"""
Example script showing how to use the PersonaChat task.

This demonstrates:
1. Loading the personachat dataset
2. Using PersonaChatDataProcessor to parse and format examples
3. Using PersonaChatEvaluator for evaluation with semantic similarity
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from transformers import AutoTokenizer
from tasks.personachat_task import PersonaChatDataProcessor, PersonaChatEvaluator


def main():
    # Load dataset
    print("Loading personachat dataset...")
    ds = load_dataset("json", data_files="personachat_train.json")
    
    # Sample a few examples
    sample = ds['train'].select(range(3))
    
    # Initialize tokenizer (using a small model for demo)
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Initialize data processor
    print("\nInitializing PersonaChatDataProcessor...")
    processor = PersonaChatDataProcessor(tokenizer)
    
    # Process examples
    print("\n" + "="*80)
    print("PROCESSING EXAMPLES")
    print("="*80)
    
    for idx, example in enumerate(sample):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {idx + 1}")
        print(f"{'='*80}")
        
        # Parse example
        parsed = processor.parse_example(example)
        
        if parsed:
            parsed_example = parsed[0]
            
            print(f"\n📋 PERSONA B (Assistant):")
            for fact in parsed_example['persona_b']:
                print(f"  - {fact}")
            
            print(f"\n💬 CONVERSATION TURNS:")
            for i, msg in enumerate(parsed_example['messages']):
                role_label = "Persona A (User)" if msg['role'] == 'user' else "Persona B (Assistant)"
                print(f"\n  [{role_label}]: {msg['content']}")
            
            print(f"\n✅ REFERENCE RESPONSE (Next from Persona B):")
            print(parsed_example['reference'])
            
            print(f"\n{'─'*80}")
            print("FORMATTED FOR TRAINING (using HF chat template):")
            print(f"{'─'*80}")
            formatted = processor.format_for_training(parsed_example)
            print(formatted)
            
            print(f"\n{'─'*80}")
            print("FORMATTED FOR INFERENCE (using HF chat template):")
            print(f"{'─'*80}")
            inference_prompt = processor.format_for_inference({
                'persona_b': parsed_example['persona_b'],
                'messages': parsed_example['messages']
            })
            print(inference_prompt)
    
    print("\n" + "="*80)
    print("EVALUATION METRICS EXAMPLE")
    print("="*80)
    
    # Example of similarity calculation
    from sentence_transformers import SentenceTransformer, util
    
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Example predictions
    ground_truth = "It's really fun! You should try it sometime."
    predictions = [
        "It's really fun! You should try it sometime.",  # Exact match
        "It's fun! You should definitely try it.",  # High similarity
        "Yes, it can be interesting.",  # Medium similarity
        "I like pizza.",  # Low similarity
    ]
    
    print("\nSemantic Similarity Scores:")
    print(f"Ground Truth: '{ground_truth}'")
    print()
    
    for pred in predictions:
        truth_emb = similarity_model.encode(ground_truth, convert_to_tensor=True)
        pred_emb = similarity_model.encode(pred, convert_to_tensor=True)
        score = util.cos_sim(truth_emb, pred_emb).item()
        
        print(f"Prediction: '{pred}'")
        print(f"Similarity Score: {score:.4f}")
        print()


if __name__ == "__main__":
    main()

