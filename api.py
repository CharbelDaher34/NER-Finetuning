"""
FastAPI app for NER inference using GGUF model.

Supports multi-turn conversations with proper chat templates.
"""
import json
import re
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_cpp import Llama
from transformers import AutoTokenizer


# Global model and tokenizer instances
model = None
tokenizer = None


# Request/Response models
class Message(BaseModel):
    """Single message in a conversation."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ConversationRequest(BaseModel):
    """Request for conversation-based inference."""
    report_text: str = Field(..., description="The crime report or document text")
    question: str = Field(..., description="Question to ask about the text")
    system_prompt: Optional[str] = Field(
        default="A virtual assistant answers questions from a user based on the provided text, answer with a json object, key being the entity asked for by user and the value extracted from the text.",
        description="System prompt for the model"
    )
    conversation_history: Optional[List[Message]] = Field(
        default=None,
        description="Previous conversation history (optional, for multi-turn conversations)"
    )
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, description="Sampling temperature")


class ConversationResponse(BaseModel):
    """Response from the model."""
    raw_response: str = Field(..., description="Raw model output")
    json_response: Dict = Field(..., description="Parsed JSON response")
    conversation_history: List[Message] = Field(..., description="Full conversation history including this turn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and tokenizer on startup, cleanup on shutdown."""
    global model, tokenizer
    
    print("="*80)
    print("Initializing NER Inference API...")
    print("="*80)
    
    # Load tokenizer for chat template
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print("✓ Tokenizer loaded")
    
    # Load GGUF model
    print("Loading GGUF model...")
    model = Llama(
        model_path="./best_model/model.gguf",
        n_ctx=40960,  # Full context capacity
        n_threads=8,
        n_gpu_layers=-1,  # Use all GPU layers
        verbose=False,
    )
    print("✓ GGUF model loaded")
    print("="*80)
    print("API Ready!")
    print("="*80)
    
    yield
    
    # Cleanup
    print("Shutting down...")
    model = None
    tokenizer = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="NER Inference API",
    description="Named Entity Recognition inference API using fine-tuned GGUF model",
    version="1.0.0",
    lifespan=lifespan
)


def clean_response(response_text: str) -> str:
    """Remove thinking tags and clean up response."""
    # Remove <think> tags - Qwen3 models generate internal reasoning
    if '<think>' in response_text:
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    
    # Clean up any stray tags
    response_text = response_text.replace('<think>', '').replace('</think>', '').strip()
    
    return response_text


def parse_json_response(response_text: str) -> Dict:
    """Parse JSON from response text."""
    try:
        # Extract JSON from response - find the first { and last }
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            return {}
        
        json_str = response_text[start_idx:end_idx+1]
        json_dict = json.loads(json_str)
        
        # Normalize values to lists
        for key, value in json_dict.items():
            if isinstance(value, list):
                pass  # Already a list
            elif isinstance(value, str):
                json_dict[key] = [value]
            elif value is not None:
                json_dict[key] = [value]
            else:
                json_dict[key] = []
        
        return json_dict
    except (json.JSONDecodeError, IndexError, KeyError, AttributeError, ValueError):
        return {}


@app.post("/infer", response_model=ConversationResponse)
async def infer_endpoint(request: ConversationRequest):
    """
    Perform NER inference on a crime report with conversation support.
    
    This endpoint supports multi-turn conversations by accepting conversation history.
    For the first question, just provide report_text and question.
    For follow-up questions, include the conversation_history from the previous response.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Build conversation messages
        messages = []
        
        # Add system prompt
        messages.append({"role": "system", "content": request.system_prompt})
        
        # If there's conversation history, add it
        if request.conversation_history:
            for msg in request.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
        else:
            # First turn: add report and acknowledgment
            messages.append({"role": "user", "content": f"Text:\n{request.report_text}"})
            messages.append({"role": "assistant", "content": "I've read this text."})
        
        # Add current question
        messages.append({"role": "user", "content": request.question})
        
        # Format using chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(prompt)
        # Generate response
        output = model(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["<|im_end|>"],
            echo=False,
        )
        
        raw_response = output['choices'][0]['text'].strip()
        
        # Clean response (remove thinking tags)
        cleaned_response = clean_response(raw_response)
        
        # Parse JSON
        json_response = parse_json_response(cleaned_response)
        
        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": cleaned_response})
        
        # Convert messages to Message objects for response
        conversation_history = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        return ConversationResponse(
            raw_response=cleaned_response,
            json_response=json_response,
            conversation_history=conversation_history
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if (model is not None and tokenizer is not None) else "initializing",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NER Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/infer": "POST - Perform NER inference with conversation support",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8347,
        reload=False  # Set to True for development
    )

