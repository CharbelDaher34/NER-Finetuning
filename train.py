import os
import re
import json
import gc
import torch
import wandb
import logging
import random
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
import shutil
from rapidfuzz import fuzz


# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Random seeds set to 42 for reproducibility")


# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
wandb_api_key = os.getenv("WANDB_API_KEY")

# Login to services
login(token=hf_token)
wandb.login(key=wandb_api_key)
run = wandb.init(
    project='Fine-Tune Llama 3 8B on Crime Dataset', 
    job_type="training", 
    anonymous="allow"
)

# Model configuration
base_model = "Qwen/Qwen3-0.6B"
new_model = "Qwen/Qwen3-0.6B-finetuned"
torch_dtype = torch.bfloat16  # Use BF16 for training
# Use SDPA (Scaled Dot Product Attention) for faster inference, fallback to eager
attn_implementation = "sdpa"  # Faster than eager, fallback to "eager" if needed
device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with LoRA (no quantization)
logger.info(f"Attempting to load model with attention implementation: {attn_implementation}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": device_id},
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    logger.info(f"✓ Successfully loaded model with {attn_implementation}")
except Exception as e:
    logger.warning(f"Failed to load with {attn_implementation}: {str(e)}")
    logger.info("Falling back to 'eager' (default attention)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": device_id},
        attn_implementation="eager",
        torch_dtype=torch_dtype,
    )
    logger.info("✓ Successfully loaded model with eager")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Sync model config with tokenizer to avoid mismatched PAD warnings
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
logger.info(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")


def parse_conversation_to_qa_pairs(conversation_text):
    """
    Parses a multi-turn conversation and extracts the report and all Q&A pairs.
    
    Returns:
        List of dicts, each containing:
        - 'system_prompt': The system message
        - 'report': The crime report text
        - 'question': A user question
        - 'answer': The assistant's answer
    """
    lines = conversation_text.strip().split('\n')
    system_prompt = lines[0]
    
    # Parse all messages
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
    
    # Extract report (first user message) and Q&A pairs
    if not messages or messages[0]["role"] != "user":
        return []
    
    # Extract report and remove "Text:" prefix if present
    report = messages[0]["content"]
    if report.startswith("Text:"):
        report = report[5:].strip()  # Remove "Text:" and any leading whitespace
    
    qa_pairs = []
    
    # Skip the first user message (report) and "I've read this text" assistant response
    # Then pair up remaining user questions with assistant answers
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


def format_chat_template(conversation_row):
    """
    Takes a row with a multi-turn conversation and converts it into separate
    training examples, one for each Q&A pair.
    
    Returns a list of formatted examples.
    """
    qa_pairs = parse_conversation_to_qa_pairs(conversation_row['conversation'])
    
    formatted_examples = []
    for qa in qa_pairs:
        # Create a conversation for each Q&A pair:
        # System -> User(report with "Text:" prefix) -> Assistant(acknowledgment) -> User(question) -> Assistant(answer)
        messages = [
            {"role": "system", "content": qa["system_prompt"]},
            {"role": "user", "content": f"Text:\n{qa['report']}"},
            {"role": "assistant", "content": "I've read this text."},
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_examples.append({"text": text})
    
    return formatted_examples


def extract_json_from_text(text):
    """
    Extracts the first valid JSON object from text using balanced brace matching.
    More robust than simple string splitting.
    
    Returns:
        dict or None: Parsed JSON object if found and valid, None otherwise
    """
    # Find the first opening brace
    start = text.find('{')
    if start == -1:
        return None
    
    # Track brace depth to find the matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # If first balanced pair fails, continue searching
                    pass
                break
    
    # No valid JSON found
    return None


def infer_using_model(report_text, question, tok, mdl, dev):
    """
    Uses a pre-trained model to process a crime report and infer answers to specific questions.
    Optimized for faster inference with robust JSON extraction.
    """
    # Stricter system prompt that enforces JSON-only output
    test_messages = [
        {"role": "system", "content": (
            "You answer ONLY with a JSON object. No prose, no backticks, nothing outside braces. "
            "Keys must match the question entity exactly. Values must be arrays of strings (or empty array)."
        )},
        {"role": "user", "content": f"Text:\n{report_text}"},
        {"role": "assistant", "content": "I've read this text."},
        {"role": "user", "content": question}
    ]

    prompt = tok.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    
    # Seed an opening brace to bias the model toward valid JSON
    prompt = prompt + "\n{"
    
    inputs = tok(prompt, return_tensors="pt", padding=True).to(dev)
    input_token_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = mdl.generate(
            **inputs, 
            max_new_tokens=200,  # Increased slightly to accommodate JSON structure
            min_new_tokens=1,
            do_sample=False,  # Greedy decoding (fastest)
            num_beams=1,  # No beam search (faster)
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,  # Enable KV cache for faster generation
        )

    new_tokens = outputs[0, input_token_length:]
    generated_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Reconstruct full response with the seeded brace
    response_text = "{" + generated_text

    # Use robust JSON extraction
    json_obj = extract_json_from_text(response_text)
    
    # Graceful fallback if extraction fails
    if json_obj is None:
        json_obj = {}
    
    # Normalize values to lists
    json_text = {}
    for key, value in json_obj.items():
        if isinstance(value, str):
            if ',' in value:
                json_text[key] = [item.strip() for item in value.split(',')]
            else:
                json_text[key] = [value] if value else []
        elif isinstance(value, list):
            json_text[key] = value
        else:
            json_text[key] = [str(value)] if value else []

    return response_text, json_text


def parse_multiturn_conversation(text):
    """
    Parses a multi-turn conversation and extracts all Q&A pairs.
    
    Returns:
        document: The crime report text
        qa_pairs: List of (question, ground_truth_json) tuples
    """
    all_blocks = re.findall(r"<\|im_start\|>(system|user|assistant)\s*(.*?)<\|im_end\|>", text, re.DOTALL)
    
    document = None
    qa_pairs = []
    current_question = None
    
    for role, content in all_blocks:
        content = content.strip()
        
        if role == "user" and document is None:
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


def is_valid_json(text):
    """Check if text is valid JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def is_schema_valid(json_obj):
    """
    Check if JSON follows the expected schema:
    {"key": [list of values]} or {"key": []}
    """
    if not isinstance(json_obj, dict):
        return False
    for value in json_obj.values():
        if not isinstance(value, list):
            return False
    return True


def calculate_metrics(predicted, ground_truth, fuzzy_threshold=85):
    """
    Calculate P/R/F1 metrics for a single prediction with fuzzy matching.
    
    Args:
        predicted: dict with keys and list values (or single values)
        ground_truth: dict with keys and list values (or single values)
        fuzzy_threshold: minimum similarity score (0-100) to consider a match
    
    Returns:
        dict with tp, fp, fn counts and fuzzy_matches list
    """
    # Normalize predicted items
    pred_items = []
    for key, values in predicted.items():
        if not isinstance(values, list):
            values = [values]
        for val in values:
            pred_items.append((key, str(val).strip().lower()))
    
    # Normalize ground truth items
    gt_items = []
    for key, values in ground_truth.items():
        if not isinstance(values, list):
            values = [values]
        for val in values:
            gt_items.append((key, str(val).strip().lower()))
    
    # Track matches
    matched_gt = set()
    matched_pred = set()
    fuzzy_matches = []
    
    # Find matches (exact + fuzzy)
    for pred_idx, (pred_key, pred_val) in enumerate(pred_items):
        best_match_score = 0
        best_match_idx = -1
        
        for gt_idx, (gt_key, gt_val) in enumerate(gt_items):
            if gt_idx in matched_gt:
                continue
                
            # Keys must match exactly
            if pred_key != gt_key:
                continue
            
            # Check if values match (exact or fuzzy)
            if pred_val == gt_val:
                # Exact match
                similarity = 100
            else:
                # Fuzzy match using token sort ratio (handles word order)
                similarity = fuzz.token_sort_ratio(pred_val, gt_val)
            
            if similarity >= fuzzy_threshold and similarity > best_match_score:
                best_match_score = similarity
                best_match_idx = gt_idx
        
        # If we found a match, mark both as matched
        if best_match_idx >= 0:
            matched_pred.add(pred_idx)
            matched_gt.add(best_match_idx)
            fuzzy_matches.append({
                "key": pred_key,
                "predicted": pred_val,
                "ground_truth": gt_items[best_match_idx][1],
                "similarity": best_match_score
            })
    
    tp = len(matched_pred)
    fp = len(pred_items) - len(matched_pred)
    fn = len(gt_items) - len(matched_gt)
    
    return {
        "tp": tp, 
        "fp": fp, 
        "fn": fn,
        "fuzzy_matches": fuzzy_matches
    }


def test_model_on_dataset(test_dataset, tok, mdl, dev):
    """
    Evaluates the model on the provided test dataset with multi-turn conversations.

    For each conversation:
      1. Extracts the document (crime report)
      2. Extracts all Q&A pairs
      3. Generates model predictions for each question
      4. Calculates metrics: P/R/F1, Set-Exact-Match, Empty-Set Accuracy, Valid-JSON%, Schema-valid%
    """
    logger.info("Starting model evaluation on test dataset")
    logger.info(f"Dataset size: {len(test_dataset)} conversations")
    
    # Optimize for evaluation: disable gradients and set model to eval mode
    mdl.eval()
    torch.set_grad_enabled(False)

    all_results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_exact_matches = 0
    total_empty_set_correct = 0
    total_empty_sets = 0
    total_valid_json = 0
    total_schema_valid = 0
    total_predictions = 0

    conversations_processed = 0
    conversations_skipped = 0

    for conv_idx, row in enumerate(tqdm(test_dataset, desc="Testing model")):
        text = row["text"]
        logger.debug(f"Processing conversation {conv_idx}")

        document, qa_pairs = parse_multiturn_conversation(text)

        if not document or not qa_pairs:
            logger.warning(f"Skipping conversation {conv_idx} (missing document or Q&A pairs)")
            conversations_skipped += 1
            continue

        conversations_processed += 1
        logger.info(f"Starting conversation {conv_idx}: Found {len(qa_pairs)} Q&A pairs to evaluate")

        for qa_idx, (question, ground_truth) in enumerate(qa_pairs):
            total_predictions += 1

            # Log minimal info for each Q&A pair
            logger.info(f"\n{'='*80}")
            logger.info(f"Conversation {conv_idx}, Q&A pair {qa_idx}")
            logger.info(f"{'='*80}")

            predicted_json = None
            response_text = ""
            is_valid_json_flag = False
            is_schema_valid_flag = False

            try:
                response_text, predicted_json = infer_using_model(document, question, tok, mdl, dev)
                is_valid_json_flag = True
                is_schema_valid_flag = is_schema_valid(predicted_json)
                logger.info("✓ Successfully generated valid prediction")
            except (json.JSONDecodeError, IndexError, KeyError, AttributeError) as e:
                predicted_json = {}
                is_valid_json_flag = False
                is_schema_valid_flag = False
                logger.error(f"✗ Error processing Q&A pair {qa_idx} in conversation {conv_idx}: {str(e)}")
                logger.error("Model JSON Output: {} (fallback due to error)")

            if is_valid_json_flag:
                total_valid_json += 1
            if is_schema_valid_flag:
                total_schema_valid += 1

            metrics = calculate_metrics(predicted_json, ground_truth, fuzzy_threshold=85)
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]

            # Check for exact match (for backwards compatibility)
            is_exact_match = (predicted_json == ground_truth)
            
            # Also check fuzzy complete match (all items matched with fuzzy)
            is_fuzzy_complete = (metrics["tp"] > 0 and metrics["fp"] == 0 and metrics["fn"] == 0)
            
            if is_exact_match:
                total_exact_matches += 1
                logger.info("✓ EXACT MATCH")
            elif is_fuzzy_complete:
                total_exact_matches += 1  # Count fuzzy complete matches as exact
                logger.info("✓ FUZZY COMPLETE MATCH")
            else:
                logger.info("✗ PARTIAL MATCH")

            # Check if empty - handle both list and non-list values
            gt_is_empty = all(
                (len(v) == 0 if isinstance(v, list) else False) 
                for v in ground_truth.values()
            )
            pred_is_empty = all(
                (len(v) == 0 if isinstance(v, list) else False) 
                for v in predicted_json.values()
            )

            if gt_is_empty:
                total_empty_sets += 1
                if pred_is_empty:
                    total_empty_set_correct += 1

            logger.info(f"Metrics: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
            logger.info(f"{'='*80}\n")

            all_results.append({
                "conversation_idx": conv_idx,
                "qa_idx": qa_idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted_json,
                "response_text": response_text,
                "exact_match": is_exact_match,
                "valid_json": is_valid_json_flag,
                "schema_valid": is_schema_valid_flag,
                "metrics": metrics
            })

            # Log progress every 50 predictions
            if total_predictions % 50 == 0:
                current_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                current_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                current_exact = total_exact_matches / total_predictions if total_predictions > 0 else 0.0
                logger.info(f"PROGRESS: Processed {total_predictions} Q&A pairs so far. "
                          f"Running metrics: P={current_precision:.3f}, R={current_recall:.3f}, Exact={current_exact:.3f}")
                # Clear cache periodically during evaluation
                gc.collect()
                torch.cuda.empty_cache()
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    set_exact_match = total_exact_matches / total_predictions if total_predictions > 0 else 0.0
    empty_set_accuracy = total_empty_set_correct / total_empty_sets if total_empty_sets > 0 else 0.0
    valid_json_pct = total_valid_json / total_predictions if total_predictions > 0 else 0.0
    schema_valid_pct = total_schema_valid / total_predictions if total_predictions > 0 else 0.0

    logger.info("Evaluation completed successfully")
    logger.info(f"Conversations processed: {conversations_processed}")
    logger.info(f"Conversations skipped: {conversations_skipped}")
    logger.info(f"Total Q&A pairs evaluated: {total_predictions}")

    logger.info("FINAL METRICS:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Set-Exact-Match: {set_exact_match:.4f}")
    logger.info(f"  Empty-Set Accuracy: {empty_set_accuracy:.4f}")
    logger.info(f"  Valid JSON %: {valid_json_pct:.4f}")
    logger.info(f"  Schema Valid %: {schema_valid_pct:.4f}")

    logger.info("Detailed counts:")
    logger.info(f"  True Positives: {total_tp}")
    logger.info(f"  False Positives: {total_fp}")
    logger.info(f"  False Negatives: {total_fn}")
    logger.info(f"  Exact Matches: {total_exact_matches}")
    logger.info(f"  Empty Sets: {total_empty_sets}")
    logger.info(f"  Empty Sets Correct: {total_empty_set_correct}")

    eval_summary = {
        "total_predictions": total_predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "set_exact_match": set_exact_match,
        "empty_set_accuracy": empty_set_accuracy,
        "valid_json_pct": valid_json_pct,
        "schema_valid_pct": schema_valid_pct,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_exact_matches": total_exact_matches,
        "total_empty_sets": total_empty_sets,
        "total_empty_set_correct": total_empty_set_correct,
        "conversations_processed": conversations_processed,
        "conversations_skipped": conversations_skipped
    }

    logger.info("Evaluation summary saved to eval_summary variable")
    return all_results, eval_summary


# Configuration for dataset processing
max_seq_length = 1536  # Optimized for 0.6B model - increased from 1024 for better context
logger.info(f"Maximum sequence length set to: {max_seq_length}")

# Load training dataset with memory mapping (doesn't load everything into RAM)
logger.info("Loading training dataset from './dataset.jsonl'")
dataset = load_dataset('json', data_files='./dataset.jsonl', keep_in_memory=False)
logger.info(f"Training dataset loaded with {len(dataset['train'])} samples (multi-turn conversations)")

# Split dataset: 90% train / 10% eval (for training validation)
logger.info("Splitting training dataset: 90% train / 10% eval")
conversation_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_conversations = conversation_split["train"]  # 90% for training
eval_conversations = conversation_split["test"]    # 10% for evaluation during training
logger.info(f"Split complete: {len(train_conversations)} conversations for training, {len(eval_conversations)} for evaluation")

# Load separate test dataset for final evaluation
logger.info("Loading test dataset from './test_dataset.jsonl'")
test_dataset = load_dataset('json', data_files='./test_dataset.jsonl', keep_in_memory=False)
test_conversations = test_dataset['train']  # The 'train' split contains all test data
logger.info(f"Test dataset loaded with {len(test_conversations)} conversations for final evaluation")


def process_conversation_batch(examples):
    """
    Process a batch of conversations on-the-fly (lazy processing).
    Splits multi-turn conversations into individual Q&A pairs.
    """
    all_texts = []
    for conversation in examples['conversation']:
        qa_pairs = parse_conversation_to_qa_pairs(conversation)
        for qa in qa_pairs:
            messages = [
                {"role": "system", "content": qa["system_prompt"]},
                {"role": "user", "content": f"Text:\n{qa['report']}"},
                {"role": "assistant", "content": "I've read this text."},
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            all_texts.append(text)
    return {"text": all_texts}


# Process training conversations: lazy map instead of loading all into memory
logger.info("Processing training dataset: setting up lazy processing for Q&A pairs")
train_data = train_conversations.map(
    process_conversation_batch,
    batched=True,
    batch_size=1,  # Process one conversation at a time
    remove_columns=train_conversations.column_names,
    desc="Processing training conversations"
)
logger.info(f"Training dataset processing completed: {len(train_data)} total training examples created")

# Process eval conversations: lazy map
logger.info("Processing eval dataset: setting up lazy processing for Q&A pairs")
eval_data = eval_conversations.map(
    process_conversation_batch,
    batched=True,
    batch_size=1,  # Process one conversation at a time
    remove_columns=eval_conversations.column_names,
    desc="Processing eval conversations"
)
logger.info(f"Eval dataset processing completed: {len(eval_data)} total eval examples created")

# Process test conversations: keep in original format for final evaluation, but add 'text' field
logger.info("Processing test dataset: formatting for final evaluation (keeping multi-turn structure)")

def format_full_conversation_for_evaluation(row):
    """Format the full multi-turn conversation for evaluation."""
    qa_pairs = parse_conversation_to_qa_pairs(row['conversation'])
    if qa_pairs:
        # Create the full multi-turn conversation with all Q&A pairs
        # Add "Text:" prefix to match inference format
        messages = [
            {"role": "system", "content": qa_pairs[0]["system_prompt"]},
            {"role": "user", "content": f"Text:\n{qa_pairs[0]['report']}"},
            {"role": "assistant", "content": "I've read this text."}
        ]
        
        # Add all Q&A pairs
        for qa in qa_pairs:
            messages.append({"role": "user", "content": qa["question"]})
            messages.append({"role": "assistant", "content": qa["answer"]})
        
        row['text'] = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        row['text'] = ""
    return row

test_data = test_conversations.map(format_full_conversation_for_evaluation)

print("\n" + "="*60)
print("DATA PREPARATION SUMMARY")
print("="*60)
print(f"Training conversations: {len(train_conversations)}")
print(f"Training examples created: {len(train_data)}")
print(f"Expansion ratio: {len(train_data) / len(train_conversations):.1f}x")
print(f"Eval conversations: {len(eval_conversations)}")
print(f"Eval examples created: {len(eval_data)}")
print(f"Test conversations (multi-turn): {len(test_data)}")
print("="*60)

print("\n--- EXAMPLE TRAINING INSTANCE ---")
print(train_data[0]['text'][:800])
print("...\n")

logger.info(f"Dataset preparation completed: {len(train_data)} train examples, {len(eval_data)} eval examples, {len(test_data)} test conversations")

# LoRA configuration (optimized for 0.6B model - increased capacity)
logger.info("Configuring LoRA parameters")
peft_config = LoraConfig(
    r=16,                   # Increased from 8 for better learning capacity on structured JSON task
    lora_alpha=32,          # Scaled proportionally (2*r)
    lora_dropout=0.05,      # Reduced from 0.1 to allow more learning
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # All attention projections
        "gate_proj", "up_proj", "down_proj"       # MLP layers
    ]
)
logger.info("LoRA configuration created (will be applied by SFTTrainer)")

# Training configuration (optimized for 0.6B model with 4k samples)
logger.info("Setting up training configuration")
sft_config = SFTConfig(
    output_dir=new_model,
    
    # Batch settings (memory-optimized)
    per_device_train_batch_size=1,        # Keep at 1 for memory efficiency
    per_device_eval_batch_size=1,         # Keep at 1 for memory efficiency
    gradient_accumulation_steps=16,       # Reduced from 32 for faster updates (effective batch size ≈ 16)
    
    # Epochs & learning rate (OPTIMIZED FOR 0.6B MODEL)
    num_train_epochs=6,                   # Increased from 3 - more epochs for 4k dataset
    learning_rate=2.4e-5,                   # Increased from 1e-5 - 0.6B can handle higher LR
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,                     # Increased from 0.05 for better stability
    
    # Precision & stability
    fp16=False,
    bf16=True,
    max_grad_norm=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Memory optimizations
    optim="adamw_torch",                  # Standard AdamW optimizer for full precision training
    weight_decay=0.001,                   # Reduced from 0.01 - less regularization for better learning
    group_by_length=True,
    
    # Logging & evaluation (more frequent for better monitoring)
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,                       # Increased frequency from 200 for better monitoring
    save_strategy="steps",
    save_steps=100,                       # Increased frequency from 200
    save_total_limit=2,                   # Increased from 1 to keep best 2 checkpoints
    load_best_model_at_end=True,          # Load best checkpoint for evaluation
    metric_for_best_model="eval_loss",    # Use eval loss to determine best model
    greater_is_better=False,              # Lower eval loss is better
    
    # Dataset details
    dataset_text_field="text",
    max_length=max_seq_length,            # Explicitly set max sequence length (now 1536)
    packing=False,
    report_to="wandb",
)
logger.info(f"Training configuration created - Scheduler: {sft_config.lr_scheduler_type}, LR: {sft_config.learning_rate}, Optimizer: {sft_config.optim}")

# Initialize trainer (without early stopping to reduce memory overhead)
logger.info("Initializing SFTTrainer")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    args=sft_config,
)
logger.info("Trainer initialized successfully")

# ✅ Reassign model to the LoRA-wrapped model from trainer
# This ensures both pre- and post-training evals use the correct model reference
model = trainer.model
logger.info("Model reference updated to use LoRA-wrapped model from trainer")

# Optional: Check for NaN parameters before training (safety check)
nan_params = []
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        nan_params.append(name)
if nan_params:
    logger.warning(f"⚠️ NaN detected in parameters: {nan_params}")
else:
    logger.info("✓ All parameters initialized correctly (no NaNs)")

# Evaluate model BEFORE training (baseline)
logger.info("\n" + "="*80)
logger.info("BASELINE EVALUATION - Model performance BEFORE training")
logger.info("="*80 + "\n")
model.config.use_cache = True
results_pre, summary_pre = test_model_on_dataset(test_data, tokenizer, model, device)
logger.info("Baseline evaluation completed")

# Clear memory after evaluation
gc.collect()
torch.cuda.empty_cache()
logger.info("Cleared CUDA cache after baseline evaluation")

# Log pre-training metrics to wandb
wandb.log({
    "pre_training/precision": summary_pre['precision'],
    "pre_training/recall": summary_pre['recall'],
    "pre_training/f1": summary_pre['f1'],
    "pre_training/set_exact_match": summary_pre['set_exact_match'],
    "pre_training/empty_set_accuracy": summary_pre['empty_set_accuracy'],
    "pre_training/valid_json_pct": summary_pre['valid_json_pct'],
    "pre_training/schema_valid_pct": summary_pre['schema_valid_pct'],
    "pre_training/total_tp": summary_pre['total_tp'],
    "pre_training/total_fp": summary_pre['total_fp'],
    "pre_training/total_fn": summary_pre['total_fn'],
    "pre_training/total_exact_matches": summary_pre['total_exact_matches'],
})

print("\n" + "="*60)
print("PRE-TRAINING EVALUATION SUMMARY")
print("="*60)
print(f"Total Predictions: {summary_pre['total_predictions']}")
print("\nCore Metrics:")
print(f"  Precision: {summary_pre['precision']:.4f}")
print(f"  Recall: {summary_pre['recall']:.4f}")
print(f"  F1 Score: {summary_pre['f1']:.4f}")
print("\nAccuracy Metrics:")
print(f"  Set-Exact-Match: {summary_pre['set_exact_match']:.4f} ({summary_pre['total_exact_matches']}/{summary_pre['total_predictions']})")
print(f"  Empty-Set Accuracy: {summary_pre['empty_set_accuracy']:.4f} ({summary_pre['total_empty_set_correct']}/{summary_pre['total_empty_sets']})")
print("\nFormat Validation:")
print(f"  Valid JSON %: {summary_pre['valid_json_pct']:.4f}")
print(f"  Schema Valid %: {summary_pre['schema_valid_pct']:.4f}")
print("="*60 + "\n")

# Train the model
logger.info("Starting training...")
# Re-enable gradients for training (disabled during evaluation)
torch.set_grad_enabled(True)
model.train()
model.config.use_cache = False
trainer.train()
logger.info("Training completed")

# Clear memory after training
gc.collect()
torch.cuda.empty_cache()
logger.info("Cleared CUDA cache after training")

# Evaluate model AFTER training
logger.info("\n" + "="*80)
logger.info("POST-TRAINING EVALUATION - Model performance AFTER training")
logger.info("="*80 + "\n")
model.config.use_cache = True
results_post, summary_post = test_model_on_dataset(test_data, tokenizer, model, device)
logger.info("Post-training evaluation completed")

# Clear memory after evaluation
gc.collect()
torch.cuda.empty_cache()
logger.info("Cleared CUDA cache after post-training evaluation")

# Log post-training metrics to wandb
wandb.log({
    "post_training/precision": summary_post['precision'],
    "post_training/recall": summary_post['recall'],
    "post_training/f1": summary_post['f1'],
    "post_training/set_exact_match": summary_post['set_exact_match'],
    "post_training/empty_set_accuracy": summary_post['empty_set_accuracy'],
    "post_training/valid_json_pct": summary_post['valid_json_pct'],
    "post_training/schema_valid_pct": summary_post['schema_valid_pct'],
    "post_training/total_tp": summary_post['total_tp'],
    "post_training/total_fp": summary_post['total_fp'],
    "post_training/total_fn": summary_post['total_fn'],
    "post_training/total_exact_matches": summary_post['total_exact_matches'],
})

# Calculate improvement
improvement = {
    "precision_delta": summary_post['precision'] - summary_pre['precision'],
    "recall_delta": summary_post['recall'] - summary_pre['recall'],
    "f1_delta": summary_post['f1'] - summary_pre['f1'],
    "set_exact_match_delta": summary_post['set_exact_match'] - summary_pre['set_exact_match'],
}

wandb.log({
    "improvement/precision_delta": improvement['precision_delta'],
    "improvement/recall_delta": improvement['recall_delta'],
    "improvement/f1_delta": improvement['f1_delta'],
    "improvement/set_exact_match_delta": improvement['set_exact_match_delta'],
})

print("\n" + "="*60)
print("POST-TRAINING EVALUATION SUMMARY")
print("="*60)
print(f"Total Predictions: {summary_post['total_predictions']}")
print("\nCore Metrics:")
print(f"  Precision: {summary_post['precision']:.4f}")
print(f"  Recall: {summary_post['recall']:.4f}")
print(f"  F1 Score: {summary_post['f1']:.4f}")
print("\nAccuracy Metrics:")
print(f"  Set-Exact-Match: {summary_post['set_exact_match']:.4f} ({summary_post['total_exact_matches']}/{summary_post['total_predictions']})")
print(f"  Empty-Set Accuracy: {summary_post['empty_set_accuracy']:.4f} ({summary_post['total_empty_set_correct']}/{summary_post['total_empty_sets']})")
print("\nFormat Validation:")
print(f"  Valid JSON %: {summary_post['valid_json_pct']:.4f}")
print(f"  Schema Valid %: {summary_post['schema_valid_pct']:.4f}")
print("="*60)

print("\n" + "="*60)
print("IMPROVEMENT SUMMARY (Post - Pre)")
print("="*60)
print(f"  Precision: {improvement['precision_delta']:+.4f}")
print(f"  Recall: {improvement['recall_delta']:+.4f}")
print(f"  F1 Score: {improvement['f1_delta']:+.4f}")
print(f"  Set-Exact-Match: {improvement['set_exact_match_delta']:+.4f}")
print("="*60)

# Create results folder with training number, 1, 2, 3, etc.
# Check last number in results folder and increment by 1
last_number = 0
try:
    for folder in os.listdir("results"):
        if folder.startswith("training_results_"):
            last_number = max(last_number, int(folder.split("_")[-1]))
    training_number = last_number + 1
except (FileNotFoundError, ValueError, IndexError):
    training_number = 1
results_folder = f"results/training_results_{training_number}"
os.makedirs(results_folder, exist_ok=True)
logger.info(f"Created results folder: {results_folder}")

# Save pre-training metrics
logger.info("Saving pre-training metrics...")
with open(os.path.join(results_folder, "metrics_pre_training.json"), 'w') as f:
    json.dump(summary_pre, f, indent=2)

with open(os.path.join(results_folder, "detailed_results_pre_training.json"), 'w') as f:
    json.dump(results_pre, f, indent=2, default=str)

# Save post-training metrics
logger.info("Saving post-training metrics...")
with open(os.path.join(results_folder, "metrics_post_training.json"), 'w') as f:
    json.dump(summary_post, f, indent=2)

with open(os.path.join(results_folder, "detailed_results_post_training.json"), 'w') as f:
    json.dump(results_post, f, indent=2, default=str)

# Save improvement deltas
logger.info("Saving improvement metrics...")
with open(os.path.join(results_folder, "metrics_improvement.json"), 'w') as f:
    json.dump(improvement, f, indent=2)

# Save training history from trainer
logger.info("Saving training history...")
training_history = {
    "train_loss_history": [log for log in trainer.state.log_history if "loss" in log],
    "final_train_loss": trainer.state.log_history[-1].get("loss", None) if trainer.state.log_history else None,
    "total_steps": trainer.state.global_step,
    "epochs_completed": trainer.state.epoch,
}
with open(os.path.join(results_folder, "training_history.json"), 'w') as f:
    json.dump(training_history, f, indent=2, default=str)

# Save training configuration
logger.info("Saving training configuration...")
training_config = {
    "base_model": base_model,
    "output_model": new_model,
    "lora_config": {
        "r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_dropout": peft_config.lora_dropout,
        "target_modules": list(peft_config.target_modules) if isinstance(peft_config.target_modules, set) else peft_config.target_modules,
    },
    "training_args": {
        "num_train_epochs": sft_config.num_train_epochs,
        "per_device_train_batch_size": sft_config.per_device_train_batch_size,
        "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
        "learning_rate": sft_config.learning_rate,
        "warmup_steps": sft_config.warmup_steps,
        "bf16": sft_config.bf16,
    },
    "dataset_info": {
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "test_samples": len(test_data),
    }
}
with open(os.path.join(results_folder, "training_config.json"), 'w') as f:
    json.dump(training_config, f, indent=2)

# Save LoRA adapter
logger.info("Saving LoRA adapter...")
adapter_folder = os.path.join(results_folder, "lora_adapter")
trainer.model.save_pretrained(adapter_folder)
tokenizer.save_pretrained(adapter_folder)
logger.info(f"LoRA adapter saved to: {adapter_folder}")

# Copy evaluation log to results folder
if os.path.exists("evaluation.log"):
    shutil.copy("evaluation.log", os.path.join(results_folder, "evaluation.log"))
    logger.info("Copied evaluation.log to results folder")

# Create summary report
logger.info("Creating summary report...")
summary_report = f"""
TRAINING SUMMARY REPORT
{'='*80}

Training Run: #{training_number}
Results Folder: {results_folder}

MODEL CONFIGURATION
{'='*80}
Base Model: {base_model}
Output Model: {new_model}

DATASET
{'='*80}
Training Samples: {len(train_data)}
Eval Samples: {len(eval_data)}
Test Samples: {len(test_data)}

PRE-TRAINING METRICS
{'='*80}
Precision:        {summary_pre['precision']:.4f}
Recall:           {summary_pre['recall']:.4f}
F1 Score:         {summary_pre['f1']:.4f}
Set-Exact-Match:  {summary_pre['set_exact_match']:.4f}
Valid JSON %:     {summary_pre['valid_json_pct']:.4f}

POST-TRAINING METRICS
{'='*80}
Precision:        {summary_post['precision']:.4f}
Recall:           {summary_post['recall']:.4f}
F1 Score:         {summary_post['f1']:.4f}
Set-Exact-Match:  {summary_post['set_exact_match']:.4f}
Valid JSON %:     {summary_post['valid_json_pct']:.4f}

IMPROVEMENT (Post - Pre)
{'='*80}
Precision:        {improvement['precision_delta']:+.4f}
Recall:           {improvement['recall_delta']:+.4f}
F1 Score:         {improvement['f1_delta']:+.4f}
Set-Exact-Match:  {improvement['set_exact_match_delta']:+.4f}

FILES SAVED
{'='*80}
- metrics_pre_training.json          : Pre-training evaluation summary
- detailed_results_pre_training.json : Detailed per-question pre-training results
- metrics_post_training.json         : Post-training evaluation summary
- detailed_results_post_training.json: Detailed per-question post-training results
- metrics_improvement.json           : Improvement deltas
- training_history.json              : Training loss history and steps
- training_config.json               : Full training configuration
- lora_adapter/                      : LoRA adapter weights and tokenizer
- evaluation.log                     : Full evaluation logs
- summary_report.txt                 : This report
"""

with open(os.path.join(results_folder, "summary_report.txt"), 'w') as f:
    f.write(summary_report)

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print(f"All metrics and model saved to: {results_folder}/")
print("\nContents:")
print("  📊 metrics_pre_training.json")
print("  📊 metrics_post_training.json")
print("  📊 metrics_improvement.json")
print("  📈 training_history.json")
print("  ⚙️  training_config.json")
print("  🤖 lora_adapter/ (model weights)")
print("  📝 evaluation.log")
print("  📄 summary_report.txt")
print("="*80)

logger.info(f"All results saved successfully to {results_folder}")
logger.info("Finishing wandb session")
wandb.finish()
logger.info("Training and evaluation pipeline completed successfully")
