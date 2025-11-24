"""
FastAPI app for NER and Persona inference using GGUF models.

Supports multi-turn conversations with proper chat templates.
"""
import json
import os
import re
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_cpp import Llama
from transformers import AutoTokenizer


# Global model and tokenizer instances
ner_model = None
persona_model = None
tokenizer = None


# Request/Response models
class Message(BaseModel):
    """Single message in a conversation."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class NERRequest(BaseModel):
    """Request for NER inference."""
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


class PersonaRequest(BaseModel):
    """Request for PersonaChat inference."""
    persona: List[str] = Field(..., description="List of persona facts defining the character")
    message: str = Field(..., description="Current user message")
    conversation_history: Optional[List[Message]] = Field(
        default=None,
        description="Previous conversation history"
    )
    max_tokens: int = Field(default=128, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")


class ConversationResponse(BaseModel):
    """Response from the model."""
    raw_response: str = Field(..., description="Raw model output")
    json_response: Optional[Dict] = Field(default=None, description="Parsed JSON response (for NER)")
    conversation_history: List[Message] = Field(..., description="Full conversation history including this turn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and tokenizer on startup, cleanup on shutdown."""
    global ner_model, persona_model, tokenizer
    
    print("="*80)
    print("Initializing Inference API...")
    print("="*80)
    
    # Load tokenizer for chat template
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise e
    
    # Load NER Model
    print("\nLoading NER model...")
    ner_path = Path("best_model/ner/model.gguf")
    if ner_path.exists():
        try:
            ner_model = Llama(
                model_path=str(ner_path.resolve()),
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=-1,
                verbose=False,
            )
            print(f"✓ NER model loaded ({ner_path})")
        except Exception as e:
            print(f"Error loading NER model: {e}")
    else:
        print(f"⚠ NER model not found at {ner_path}")

    # Load Persona Model
    print("\nLoading Persona model...")
    persona_path = Path("best_model/personachat/model.gguf")
    if persona_path.exists():
        try:
            persona_model = Llama(
                model_path=str(persona_path.resolve()),
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=-1,
                verbose=False,
            )
            print(f"✓ Persona model loaded ({persona_path})")
        except Exception as e:
            print(f"Error loading Persona model: {e}")
    else:
        print(f"⚠ Persona model not found at {persona_path}")
    
    print("="*80)
    print("API Ready!")
    print("="*80)
    
    yield
    
    # Cleanup
    print("Shutting down...")
    ner_model = None
    persona_model = None
    tokenizer = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Multi-Task Inference API",
    description="Inference API for NER and PersonaChat using fine-tuned GGUF models",
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


@app.post("/ner", response_model=ConversationResponse)
async def ner_endpoint(request: NERRequest):
    """
    Perform NER inference on a crime report.
    """
    if ner_model is None:
        raise HTTPException(status_code=503, detail="NER model not loaded")
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        # Build conversation messages
        messages = []
        messages.append({"role": "system", "content": request.system_prompt})
        
        if request.conversation_history:
            for msg in request.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
        else:
            messages.append({"role": "user", "content": f"Text:\n{request.report_text}"})
            messages.append({"role": "assistant", "content": "I've read this text."})
        
        messages.append({"role": "user", "content": request.question})
        
        # Format prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        output = ner_model(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["<|im_end|>"],
            echo=False,
        )
        
        raw_response = output['choices'][0]['text'].strip()
        cleaned_response = clean_response(raw_response)
        json_response = parse_json_response(cleaned_response)
        
        messages.append({"role": "assistant", "content": cleaned_response})
        
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
        raise HTTPException(status_code=500, detail=f"NER Inference error: {str(e)}")


@app.post("/persona", response_model=ConversationResponse)
async def persona_endpoint(request: PersonaRequest):
    """
    Perform PersonaChat inference.
    """
    if persona_model is None:
        raise HTTPException(status_code=503, detail="Persona model not loaded")
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        # Build system prompt from persona
        persona_text = "\n".join([f"- {fact}" for fact in request.persona])
        system_prompt = (
            "You are a conversational AI with the following persona:\n"
            f"{persona_text}\n\n"
            "Respond naturally and stay true to your persona."
        )
        
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        
        if request.conversation_history:
            for msg in request.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": request.message})
        
        # Format prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        output = persona_model(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["<|im_end|>"],
            echo=False,
        )
        
        raw_response = output['choices'][0]['text'].strip()
        cleaned_response = clean_response(raw_response)
        
        messages.append({"role": "assistant", "content": cleaned_response})
        
        conversation_history = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        return ConversationResponse(
            raw_response=cleaned_response,
            json_response=None,
            conversation_history=conversation_history
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Persona Inference error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ner_model_loaded": ner_model is not None,
        "persona_model_loaded": persona_model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multi-Task Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/ner": "POST - Named Entity Recognition",
            "/persona": "POST - PersonaChat Conversation",
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
        reload=False
    )

