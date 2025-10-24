"""
Test inference on one row from test_dataset.jsonl
"""
import json
from inference_gguf import infer

# Read first row from test dataset
with open('test_dataset.jsonl', 'r') as f:
    first_row = json.loads(f.readline())

# Extract the conversation
conversation = first_row['conversation']

# Extract the crime report text from the conversation
# The text is in the first User message after "Text:\n"
lines = conversation.split('\n')
text_start = False
crime_text = []

for line in lines:
    if line.startswith('User: Text:'):
        text_start = True
        continue
    elif text_start:
        if line.startswith('Assistant:'):
            break
        crime_text.append(line)

crime_report = '\n'.join(crime_text).strip()

print("=" * 80)
print("CRIME REPORT TEXT:")
print("=" * 80)
print(crime_report)
print("\n" + "=" * 80)

# Setup for inference
system_prompt = "A virtual assistant answers questions from a user based on the provided text, answer with a json object, key being the entity asked for by user and the value extracted from the text."
question = "What describes Location in the text?"

print("INFERENCE SETUP:")
print("=" * 80)
print(f"System Prompt: {system_prompt[:100]}...")
print(f"Question: {question}")
print(f"Report length: {len(crime_report)} characters")
print("\n" + "=" * 80)

# Run inference
print("RUNNING INFERENCE...")
print("=" * 80)

try:
    response_text, json_response = infer(
        model_path="results/training_results_20251024_071121/model.gguf",
        system_prompt=system_prompt,
        report_text=crime_report,
        question=question,
        max_tokens=200,
        temperature=0.1
    )
    
    print("\nRAW MODEL RESPONSE:")
    print("=" * 80)
    print(response_text)
    print("=" * 80)
    
    print("\nPARSED JSON RESPONSE:")
    print("=" * 80)
    print(json.dumps(json_response, indent=2))
    print("=" * 80)
    
    # Show expected response
    print("\nEXPECTED RESPONSE:")
    print("=" * 80)
    # Find the expected answer in the conversation
    if '"Location":' in conversation:
        start_idx = conversation.find('"Location":')
        end_idx = conversation.find('\n', start_idx)
        expected = conversation[start_idx:end_idx]
        print(expected)
    print("=" * 80)
    
except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()

