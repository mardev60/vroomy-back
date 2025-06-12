from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn

app = FastAPI(
    title="Car Price Prediction API",
    description="API pour prédire les prix des voitures",
    version="1.0.0"
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

# Chargement du modèle au démarrage
print("Chargement du modèle...")

# Configuration
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./car_price_model_final"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Chargement du modèle de base
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Chargement des adaptateurs LoRA
model = PeftModel.from_pretrained(
    model,
    adapter_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Modèle chargé avec succès! Utilisation de : {device}")

def generate_prediction(prompt: str) -> str:
    # Préparation du prompt
    input_text = f"<|system|>You are a car price prediction assistant.</s><|user|>{prompt}</s><|assistant|>"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Génération de la réponse
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
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
        "model": "loaded",
        "device": device
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False) 