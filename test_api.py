"""
Test client for the Multi-Task Inference API.

Tests both NER and PersonaChat endpoints.
"""
import requests
import json


API_URL = "http://localhost:8347"

# Sample crime report for NER
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

# Sample persona for PersonaChat
PERSONA = [
    "I am an artist.",
    "I love painting landscapes.",
    "I live in a small studio in Paris.",
    "I enjoy drinking coffee at sidewalk cafes."
]


def test_health():
    """Test health check endpoint."""
    print("\n" + "="*80)
    print("Testing Health Check")
    print("="*80)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_ner_single_turn():
    """Test NER single-turn conversation."""
    print("\n" + "="*80)
    print("Testing NER Single-Turn")
    print("="*80)
    
    payload = {
        "report_text": CRIME_REPORT,
        "question": "What describes Location in the text?",
        "max_tokens": 512,
        "temperature": 0.0
    }
    
    print("\nSending request to /ner...")
    response = requests.post(f"{API_URL}/ner", json=payload)
    
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


def test_ner_multi_turn():
    """Test NER multi-turn conversation."""
    print("\n" + "="*80)
    print("Testing NER Multi-Turn")
    print("="*80)
    
    # First question
    print("\n--- Question 1: Location ---")
    payload = {
        "report_text": CRIME_REPORT,
        "question": "What describes Location in the text?",
        "max_tokens": 512,
        "temperature": 0.0
    }
    
    response = requests.post(f"{API_URL}/ner", json=payload)
    
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
        "report_text": CRIME_REPORT,
        "question": "What describes Officer_Name in the text?",
        "conversation_history": conversation_history,
        "max_tokens": 512,
        "temperature": 0.0
    }
    
    response = requests.post(f"{API_URL}/ner", json=payload)
    
    if response.status_code != 200:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    print("Answer:", json.dumps(result["json_response"], indent=2))
    
    print("\n✓ NER Multi-turn conversation completed!")


def test_persona_chat():
    """Test PersonaChat conversation."""
    print("\n" + "="*80)
    print("Testing PersonaChat")
    print("="*80)
    
    print("Persona:")
    for p in PERSONA:
        print(f"- {p}")
    
    # Turn 1
    message1 = "I have all artists and paris and coffess"
    print(f"\nUser: {message1}")
    
    payload = {
        "persona": PERSONA,
        "message": message1,
        "max_tokens": 128,
        "temperature": 0.7
    }
    
    response = requests.post(f"{API_URL}/persona", json=payload)
    
    if response.status_code != 200:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    response1 = result["raw_response"]
    print(f"Assistant: {response1}")
    
    conversation_history = result["conversation_history"]
    
    # Turn 2
    message2 = "That sounds lovely. Do you have a favorite place to paint?"
    print(f"\nUser: {message2}")
    
    payload = {
        "persona": PERSONA,
        "message": message2,
        "conversation_history": conversation_history,
        "max_tokens": 128,
        "temperature": 0.7
    }
    
    response = requests.post(f"{API_URL}/persona", json=payload)
    
    if response.status_code != 200:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    response2 = result["raw_response"]
    print(f"Assistant: {response2}")
    
    print("\n✓ PersonaChat conversation completed!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Multi-Task API Test Client")
    print("="*80)
    print(f"API URL: {API_URL}")
    print("\nMake sure the API is running:")
    print("  uv run api.py")
    print("="*80)
    
    try:
        # Test endpoints
        test_health()
        # test_ner_single_turn()
        # test_ner_multi_turn()
        test_persona_chat()
        
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

