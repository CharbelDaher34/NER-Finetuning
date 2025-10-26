# NER Fine-Tuning Project

Fine-tune Qwen3-0.6B for Named Entity Recognition on crime reports with LoRA, convert to GGUF, and serve via FastAPI.

## Features

- Multi-turn conversational training with Q&A format
- LoRA-based fine-tuning for memory efficiency
- Fuzzy matching evaluation (85% similarity threshold)
- GGUF conversion for optimized inference
- FastAPI REST endpoint

## Installation

```bash
# Clone and navigate to project
git clone <repository-url>
cd NER-Finetuning

# Install dependencies
uv add torch transformers datasets peft trl wandb huggingface-hub python-dotenv rapidfuzz tqdm numpy
uv add fastapi uvicorn pydantic llama-cpp-python

# Create .env file
echo "HF_TOKEN=your_huggingface_token" > .env
echo "WANDB_API_KEY=your_wandb_api_key" >> .env
```

## Dataset Format

JSONL format with multi-turn conversations. Each line:

```json
{
  "conversation": "You are an assistant that extracts entities from crime reports.\nUser: Text:\n[Crime report text]\nAssistant: I've read this text.\nUser: Extract all locations mentioned.\nAssistant: {\"locations\": [\"Main Street\", \"Downtown\"]}\nUser: Extract all dates mentioned.\nAssistant: {\"dates\": [\"January 15, 2024\"]}"
}
```

**Structure:**
1. System prompt with instructions
2. User: `Text:\n[document]`
3. Assistant: `I've read this text.`
4. User: Question about entities
5. Assistant: JSON response `{"entity_type": ["value1", "value2"]}`

**Required files:**
- `dataset.jsonl` - Training data
- `test_dataset.jsonl` - Test data

## Usage

### Step 1: Train the Model

```bash
uv run train.py
```

**Outputs:** `results/training_results_N/`
- `lora_adapter/` - Model weights
- `metrics_*.json` - Evaluation results
- `training_history.json` - Training logs
- `summary_report.txt` - Summary

**Training Config:**
- Batch size: 1, Gradient accumulation: 16
- Learning rate: 2.4e-5, LoRA rank: 16
- Max sequence length: 1536, Epochs: 6

### Step 2: Convert to GGUF

```bash
uv run convert_to_gguf.py
```

Merges LoRA adapter with base model and converts to GGUF (Q4_K_M quantization).

### Step 3: Run the API

```bash
uv run api.py
```

API runs at `http://localhost:8000`

**Example request:**
```bash
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "On January 15, 2024, officers responded to 123 Main Street.",
    "question": "Extract all dates and locations mentioned."
  }'
```

**Response:**
```json
{
  "entities": {
    "dates": ["January 15, 2024"],
    "locations": ["123 Main Street"]
  },
  "raw_response": "{\"dates\": [\"January 15, 2024\"], \"locations\": [\"123 Main Street\"]}"
}
```

## Evaluation Metrics

- **Precision/Recall/F1**: Entity extraction accuracy with fuzzy matching
- **Set-Exact-Match**: Percentage of perfect predictions
- **Empty-Set Accuracy**: Accuracy when no entities present
- **Valid JSON %**: Percentage of valid JSON responses
- **Schema Valid %**: Percentage following expected schema

## Model Configuration

- **Base Model**: Qwen/Qwen3-0.6B
- **Precision**: BF16
- **LoRA**: r=16, alpha=32, dropout=0.05
- **Targets**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Troubleshooting

**OOM Errors:**
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`

**Poor Performance:**
- Increase `num_train_epochs` or LoRA rank
- Check dataset quality and JSON formatting
- Adjust `learning_rate`

**GGUF Conversion:**
- Verify llama.cpp installation
- Check LoRA adapter path in script
- Ensure sufficient disk space

## Project Structure

```
NER-Finetuning/
├── train.py              # Training script
├── convert_to_gguf.py    # GGUF conversion
├── api.py                # FastAPI server
├── inference_gguf.py     # GGUF inference
├── dataset.jsonl         # Training data
├── test_dataset.jsonl    # Test data
├── .env                  # Environment variables
└── results/              # Training outputs
```
