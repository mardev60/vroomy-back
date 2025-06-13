from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn
from functools import lru_cache
import os
import multiprocessing

# Configuration
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./car_price_model_final"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration du nombre de workers
NUM_WORKERS = min(8, multiprocessing.cpu_count())  # Utilise jusqu'à 8 workers

# Configuration de la mémoire
torch.cuda.empty_cache() if torch.cuda.is_available() else None
torch.set_num_threads(NUM_WORKERS)

app = FastAPI(
    title="Car Price Prediction API",
    description="API pour prédire les prix des voitures",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les origines exactes
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes
    allow_headers=["*"],  # Permet tous les headers
    expose_headers=["*"],
    max_age=3600,
)

class PredictionRequest(BaseModel):
    prompt: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Je recherche une Volkswagen Golf de 2015, modèle manuel avec un moteur 2.0L diesel. Elle a 50000 miles au compteur, consomme 60 mpg et a une taxe annuelle de 145 £. Quel est son prix ?"
            }
        }

class PredictionResponse(BaseModel):
    price_prediction: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "price_prediction": "Le prix de cette voiture est de 12499 £."
            }
        }

@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        num_workers=NUM_WORKERS
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@lru_cache(maxsize=1)
def get_model():
    # Configuration pour optimiser l'utilisation de la mémoire
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "max_memory": {0: "6GB"} if torch.cuda.is_available() else None
    }
    
    # Chargement du modèle de base
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs
    )
    
    # Chargement des adaptateurs LoRA
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Optimisation de la mémoire
    model.eval()
    return model

def generate_prediction(prompt: str) -> str:
    # Chargement lazy du tokenizer et du modèle
    tokenizer = get_tokenizer()
    model = get_model()
    
    # Préparation du prompt
    input_text = f"<|system|>You are a car price prediction assistant.</s><|user|>{prompt}</s><|assistant|>"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Génération de la réponse avec des paramètres optimisés
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,  # Réduit l'utilisation de la mémoire
            use_cache=True
        )
    
    # Décodage et nettoyage de la réponse
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Nettoyage de la réponse
    response = response.replace(input_text, "")
    response = response.replace("<|system|>You are a car price prediction assistant.", "")
    response = response.replace("<|user|>", "")
    response = response.replace("<|assistant|>", "")
    response = response.replace("</s>", "")
    return response.strip()

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Prédit le prix d'une voiture basé sur sa description.
    """
    try:
        prediction = generate_prediction(request.prompt)
        return PredictionResponse(price_prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Vérifie que l'API est en ligne et que le modèle est chargé.
    """
    return {
        "status": "healthy",
        "model": "ready",
        "device": device,
        "workers": NUM_WORKERS,
        "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f}MB" if torch.cuda.is_available() else "N/A"
    }

if __name__ == "__main__":
    # Configuration du serveur pour utiliser les workers
    config = uvicorn.Config(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=NUM_WORKERS,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run() 