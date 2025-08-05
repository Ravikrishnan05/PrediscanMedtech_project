# --- Import necessary libraries ---
import uvicorn  # Used to run the FastAPI server
import torch  # PyTorch for model operations
import io  # For converting byte stream to image
import traceback  # To print errors if anything goes wrong

from fastapi import FastAPI, File, UploadFile, Form  # FastAPI modules for HTTP API
from fastapi.responses import JSONResponse  # To send responses in JSON
from fastapi.middleware.cors import CORSMiddleware  # To allow access from any frontend
from PIL import Image  # To handle image uploads
from unsloth import FastLanguageModel  # Loads and optimizes language model (Unsloth is a wrapper)
from transformers import AutoProcessor  # Preprocessing: text + image -> model input

# --- Load AI model and processor (image+text) ---
print("🚀 [API] Loading model...")

model_id = "google/medgemma-4b-it"  # The model you are using (Google’s Med-GEMMA model)

# Load model and tokenizer with optimizations (4-bit quantization, float16)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Load processor separately (this handles text + image input formatting)
processor = AutoProcessor.from_pretrained(model_id)

# --- Setup FastAPI web server ---
app = FastAPI()

# Allow CORS for all origins (helps frontend talk to backend from different URLs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Root endpoint: just to check if server is alive ---
@app.get("/")
def read_root():
    return {"message": "✅ API Backend is running!"}

# --- Main endpoint: receives image + prompt, returns AI-generated response ---
@app.post("/generate")
async def generate(prompt: str = Form(...), image: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Prepare prompt for multi-modal (image + text) processing
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]

        # Convert chat messages into a model-readable string
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Prepare inputs (convert image + text to tensor)
        inputs = processor(text=prompt_text, images=pil_image, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)  # Move to GPU

        # Generate response using the model
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode the output tokens into readable text
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Clean up response by removing prefixes like "Assistant:"
        response_text = generated_texts[0].split("Assistant: ")[-1].strip()

        # Return JSON response
        return JSONResponse(content={"response": response_text})

    except Exception as e:
        # If any error happens, return a 500 with error info
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Run server (for local testing or debugging only) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
