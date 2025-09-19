from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import pipeline, AutoTokenizer
from typing import Optional
import threading
import os

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>MedGemma Chat API</h1><p>Frontend not found.</p>", status_code=200)

# Global state for lazy loading
app.state.text_pipe = None  # type: Optional[object]
app.state.tokenizer = None  # type: Optional[AutoTokenizer]
app.state.load_lock = threading.Lock()

def get_model_id() -> str:
    # Configure via env. Set to your MedGemma repo if you have access.
    # e.g. MODEL_ID="google/med-gemma-2b-it" (replace with exact repo id)
    # Default to a small instruction-tuned chat model for good UX on greetings.
    return os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def ensure_pipeline_loaded() -> None:
    if app.state.text_pipe is not None:
        return
    with app.state.load_lock:
        if app.state.text_pipe is not None:
            return
        model_id = get_model_id()
        # trust_remote_code can be needed for some repos; safe default is False
        # Load tokenizer for chat template support
        app.state.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        app.state.text_pipe = pipeline(
            task="text-generation",
            model=model_id,
            tokenizer=app.state.tokenizer,
            # device_map="auto"  # Uncomment if you use accelerate and want auto placement
        )

@app.get("/health")
async def health():
    status = {
        "status": "ok",
        "model_id": get_model_id(),
        "model_loaded": app.state.text_pipe is not None,
    }
    return JSONResponse(status)

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return JSONResponse({"error": "Empty message."}, status_code=400)

        ensure_pipeline_loaded()
        pipe = app.state.text_pipe
        tok = app.state.tokenizer
        if pipe is None:
            return JSONResponse({"error": "Model not loaded."}, status_code=503)

        system_prompt = os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful, concise assistant. Answer briefly and clearly.",
        )

        prompt_text: str
        if tok is not None and hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = f"System: {system_prompt}\nUser: {user_message}\nAssistant:"

        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "128"))
        temperature = float(os.getenv("TEMPERATURE", "0.2"))
        top_p = float(os.getenv("TOP_P", "0.9"))
        repetition_penalty = float(os.getenv("REPETITION_PENALTY", "1.05"))

        outputs = pipe(
            prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=False if temperature == 0 else True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        full_text = outputs[0].get("generated_text", "") if outputs else ""
        # Strip the prompt to return only the assistant continuation
        reply = full_text[len(prompt_text) :] if full_text.startswith(prompt_text) else full_text
        return JSONResponse({"response": reply.strip()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
