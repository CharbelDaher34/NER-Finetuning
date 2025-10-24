"""
Test client for the NER Inference API.

Demonstrates single-turn and multi-turn conversations.
"""
import requests
import json


API_URL = "http://localhost:8347"

# Sample crime report
CRIME_REPORT = """**Crime Type:** Theft  
**Date and Time:** September 30, 2025, at 14:30  
**Location:** 455 Elm Street, Cedar Falls  
**Reporting Officer:** Officer Sarah Walden, Badge #4387  
**Summary:** A burglary was reported at a local electronics store, resulting in the theft of several high-value items.  
**Description of Victim(s):** Tech Haven, owned by Mr. Allen Brigs, age 45.  
**Description of Suspect(s) (if applicable):** Not provided  
**Witnesses (if any):** None identified  
**Evidence Collected:** Surveillance footage, fingerprints  
**Circumstances Surrounding the Incident:** The store was closed for lunch when the suspect reportedly broke a side window and entered the premises. The owner discovered the break-in upon reopening at 15:00 and promptly contacted the authorities.  
**Initial Investigation:** Officer Walden arrived on the scene, reviewed the surveillance footage, and collected fingerprints from the window.  
**Further Steps:** Detectives will analyze the collected evidence and interview nearby businesses for additional information.  
**Current Status:** Under Investigation  
**Conclusion:** The case remains open as authorities continue to pursue leads related to the theft.  
**Signature:** Officer Sarah Walden"""


def test_health():
    """Test health check endpoint."""
    print("\n" + "="*80)
    print("Testing Health Check")
    print("="*80)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_single_turn():
    """Test single-turn conversation (one question)."""
    print("\n" + "="*80)
    print("Testing Single-Turn Conversation")
    print("="*80)
    
    payload = {
        "report_text": CRIME_REPORT,
        "question": "What describes Location in the text?",
        "max_tokens": 512,
        "temperature": 0.0
    }
    
    print("\nSending request...")
    response = requests.post(f"{API_URL}/infer", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✓ Success!")
        print("\nJSON Response:")
        print(json.dumps(result["json_response"], indent=2))
        print("\nRaw Response:")
        print(result["raw_response"])
        
        return result["conversation_history"]
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)
        return None


def test_multi_turn():
    """Test multi-turn conversation (multiple questions)."""
    print("\n" + "="*80)
    print("Testing Multi-Turn Conversation")
    print("="*80)
    
    # First question
    print("\n--- Question 1: Location ---")
    payload = {
        "report_text": CRIME_REPORT,
        "question": "What describes Location in the text?",
        "max_tokens": 512,
        "temperature": 0.0
    }
    
    response = requests.post(f"{API_URL}/infer", json=payload)
    
    if response.status_code != 200:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    print("Answer:", json.dumps(result["json_response"], indent=2))
    
    # Save conversation history for next question
    conversation_history = result["conversation_history"]
    
    # Second question - using conversation history
    print("\n--- Question 2: Officer Name ---")
    payload = {
        "report_text": CRIME_REPORT,  # Still needed for reference
        "question": "What describes Officer_Name in the text?",
        "conversation_history": conversation_history,  # Include previous turns
        "max_tokens": 512,
        "temperature": 0.0
    }
    
    response = requests.post(f"{API_URL}/infer", json=payload)
    
    if response.status_code != 200:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    print("Answer:", json.dumps(result["json_response"], indent=2))
    
    # Update conversation history
    conversation_history = result["conversation_history"]
    
    # Third question
    print("\n--- Question 3: Crime Type ---")
    payload = {
        "report_text": CRIME_REPORT,
        "question": "What describes Crime_Type in the text?",
        "conversation_history": conversation_history,
        "max_tokens": 512,
        "temperature": 0.0
    }
    
    response = requests.post(f"{API_URL}/infer", json=payload)
    
    if response.status_code != 200:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    print("Answer:", json.dumps(result["json_response"], indent=2))
    
    print("\n✓ Multi-turn conversation completed!")
    print(f"Total conversation turns: {len(result['conversation_history'])}")


def test_batch_questions():
    """Test extracting multiple entities."""
    print("\n" + "="*80)
    print("Testing Batch Entity Extraction")
    print("="*80)
    
    questions = [
        "What describes Location in the text?",
        "What describes Officer_Name in the text?",
        "What describes Officer_BadgeNumber in the text?",
        "What describes Victim_Name in the text?",
        "What describes Crime_Type in the text?",
    ]
    
    results = {}
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        
        payload = {
            "report_text": CRIME_REPORT,
            "question": question,
            "max_tokens": 512,
            "temperature": 0.0
        }
        
        response = requests.post(f"{API_URL}/infer", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            results.update(result["json_response"])
            print("  ✓", json.dumps(result["json_response"], indent=2))
        else:
            print(f"  ✗ Error: {response.status_code}")
    
    print("\n" + "="*80)
    print("All Extracted Entities:")
    print("="*80)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    print("\n" + "="*80)
    print("NER API Test Client")
    print("="*80)
    print(f"API URL: {API_URL}")
    print("\nMake sure the API is running:")
    print("  uv run api.py")
    print("="*80)
    
    try:
        # Test endpoints
        # test_health()
        # test_single_turn()
        test_multi_turn()
        test_batch_questions()
        
        print("\n" + "="*80)
        print("All tests completed!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("Make sure the API is running: uv run api.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

