import os
import re
import json
import torch
import wandb
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login


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
torch_dtype = torch.float16
attn_implementation = "eager"
device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"": device_id},
    attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_chat_template(row):
    """
    Parses a conversation from a single string into a list of dictionaries
    with "role" and "content" keys.
    """
    full_conversation = row['conversation']
    lines = full_conversation.strip().split('\n')
    system_prompt = lines[0]
    messages = [{"role": "system", "content": system_prompt}]
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
    
    row['text'] = tokenizer.apply_chat_template(messages, tokenize=False)
    return row


def infer_using_model(report_text, question, tok, mdl, dev):
    """
    Uses a pre-trained model to process a crime report and infer answers to specific questions.
    """
    test_messages = [
        {"role": "system", "content": "A virtual assistant answers questions from a user based on the provided text, answer with a json object, key being the entity asked for by user and the value extracted from the text."},
        {"role": "user", "content": f"Text:\n{report_text}"},
        {"role": "assistant", "content": "I've read this text."},
        {"role": "user", "content": question}
    ]

    prompt = tok.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", padding=True).to(dev)

    input_token_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = mdl.generate(**inputs, max_new_tokens=500, do_sample=False)

    new_tokens = outputs[0, input_token_length:]
    response_text = tok.decode(new_tokens, skip_special_tokens=True)

    json_text = json.loads(f"{{{response_text.split('{')[1].split('}')[0].strip()}}}")

    for key, value in json_text.items():
        if isinstance(value, str) and ',' in value:
            json_text[key] = [item.strip() for item in value.split(',')]
        elif isinstance(value, str):
            json_text[key] = [value]

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


def calculate_metrics(predicted, ground_truth):
    """
    Calculate P/R/F1 metrics for a single prediction.
    
    Args:
        predicted: dict with keys and list values
        ground_truth: dict with keys and list values
    
    Returns:
        dict with tp, fp, fn counts
    """
    pred_items = set()
    gt_items = set()
    
    for key, values in predicted.items():
        for val in values:
            pred_items.add((key, str(val)))
    
    for key, values in ground_truth.items():
        for val in values:
            gt_items.add((key, str(val)))
    
    tp = len(pred_items & gt_items)
    fp = len(pred_items - gt_items)
    fn = len(gt_items - pred_items)
    
    return {"tp": tp, "fp": fp, "fn": fn}


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

            # Log the actual content for each Q&A pair
            logger.info(f"\n{'='*80}")
            logger.info(f"Conversation {conv_idx}, Q&A pair {qa_idx}")
            logger.info(f"{'='*80}")
            logger.info(f"Report: {document[:200]}{'...' if len(document) > 200 else ''}")
            logger.info(f"User Question: {question}")
            logger.info(f"Ground Truth: {json.dumps(ground_truth, indent=2)}")

            predicted_json = None
            response_text = ""
            is_valid_json_flag = False
            is_schema_valid_flag = False

            try:
                response_text, predicted_json = infer_using_model(document, question, tok, mdl, dev)
                is_valid_json_flag = True
                is_schema_valid_flag = is_schema_valid(predicted_json)
                logger.info(f"Model JSON Output: {json.dumps(predicted_json, indent=2)}")
                logger.info("✓ Successfully generated valid prediction")
            except (json.JSONDecodeError, IndexError, KeyError, AttributeError) as e:
                predicted_json = {}
                is_valid_json_flag = False
                is_schema_valid_flag = False
                logger.error(f"✗ Error processing Q&A pair {qa_idx} in conversation {conv_idx}: {str(e)}")
                logger.error(f"Raw Model Response: {response_text[:500]}{'...' if len(response_text) > 500 else ''}")
                logger.error(f"Fallback Ground Truth: {json.dumps(ground_truth, indent=2)}")
                logger.error("Model JSON Output: {} (fallback due to error)")

            if is_valid_json_flag:
                total_valid_json += 1
            if is_schema_valid_flag:
                total_schema_valid += 1

            metrics = calculate_metrics(predicted_json, ground_truth)
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]

            is_exact_match = (predicted_json == ground_truth)
            if is_exact_match:
                total_exact_matches += 1
                logger.info("✓ EXACT MATCH - Prediction matches ground truth perfectly")
            else:
                logger.info("✗ PARTIAL MATCH - Some differences found")
                # Show what was different
                for key in set(ground_truth.keys()) | set(predicted_json.keys()):
                    gt_val = ground_truth.get(key, [])
                    pred_val = predicted_json.get(key, [])
                    if gt_val != pred_val:
                        logger.info(f"  Difference in '{key}': GT={gt_val} vs Pred={pred_val}")

            gt_is_empty = all(len(v) == 0 for v in ground_truth.values())
            pred_is_empty = all(len(v) == 0 for v in predicted_json.values())

            if gt_is_empty:
                total_empty_sets += 1
                if pred_is_empty:
                    total_empty_set_correct += 1
                    logger.info("✓ Correctly identified empty set")
                else:
                    logger.info("✗ False positive - predicted values for empty ground truth")
            else:
                if pred_is_empty:
                    logger.info("✗ False negative - missed non-empty ground truth")

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


# Load and process dataset
logger.info("Loading dataset from './dataset.jsonl'")
dataset = load_dataset('json', data_files='./dataset.jsonl')
logger.info(f"Dataset loaded with {len(dataset['train'])} samples")

train_dataset = dataset['train'].select(range(20))
logger.info(f"Selected first 10 samples for training: {len(train_dataset)} samples")

logger.info("Processing dataset with chat template formatting")
processed_dataset = train_dataset.map(format_chat_template, num_proc=2)
logger.info("Dataset processing completed")

print("--- ORIGINAL CONVERSATION STRING ---")
print(processed_dataset[0]['conversation'])
print("\n" + "="*50 + "\n")
print("--- PROCESSED AND TEMPLATED TEXT ---")
print(processed_dataset[0]['text'])

# Split dataset
logger.info("Splitting dataset into train/test sets")
data_split = processed_dataset.train_test_split(test_size=0.1, seed=42)
train_data = data_split["train"]
test_data = data_split["test"]
logger.info(f"Dataset split completed: {len(train_data)} train samples, {len(test_data)} test samples")

# LoRA configuration
logger.info("Configuring LoRA parameters")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj']
)
logger.info("Applying LoRA to model")
model = get_peft_model(model, peft_config)
logger.info("LoRA configuration applied successfully")

# Training configuration
logger.info("Setting up training configuration")
sft_config = SFTConfig(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    logging_steps=1,
    warmup_steps=2,
    learning_rate=1e-4,
    fp16=True,
    bf16=False,
    group_by_length=False,
    report_to="wandb",
    dataset_text_field="text",
    packing=False,
)
logger.info("Training configuration created")

# Initialize trainer
logger.info("Initializing SFTTrainer")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=peft_config,
    args=sft_config,
)
logger.info("Trainer initialized successfully")

# Uncomment to train
# logger.info("Starting training...")
# trainer.train()
# logger.info("Training completed")

logger.info("Finishing wandb session")
wandb.finish()
model.config.use_cache = True
logger.info("Model cache enabled")

# Test model on dataset
logger.info("Starting model evaluation on test dataset")
results, summary = test_model_on_dataset(test_data, tokenizer, model, device)
logger.info("Model evaluation completed")

print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"Total Predictions: {summary['total_predictions']}")
print("\nCore Metrics:")
print(f"  Precision: {summary['precision']:.4f}")
print(f"  Recall: {summary['recall']:.4f}")
print(f"  F1 Score: {summary['f1']:.4f}")
print("\nAccuracy Metrics:")
print(f"  Set-Exact-Match: {summary['set_exact_match']:.4f} ({summary['total_exact_matches']}/{summary['total_predictions']})")
print(f"  Empty-Set Accuracy: {summary['empty_set_accuracy']:.4f} ({summary['total_empty_set_correct']}/{summary['total_empty_sets']})")
print("\nFormat Validation:")
print(f"  Valid JSON %: {summary['valid_json_pct']:.4f}")
print(f"  Schema Valid %: {summary['schema_valid_pct']:.4f}")
print("\nDetailed Counts:")
print(f"  True Positives: {summary['total_tp']}")
print(f"  False Positives: {summary['total_fp']}")
print(f"  False Negatives: {summary['total_fn']}")
print("="*60)
