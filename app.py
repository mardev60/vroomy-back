from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from peft import LoraConfig, PeftModel

app = FastAPI(title="Model API", description="API pour le modèle fine-tuné avec LoRA")

# Configuration
BASE_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "car_price_model_final"

def load_adapter_config():
    config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

try:
    print("Chargement du modèle...")
    
    # Charger le tokenizer depuis le modèle de base
    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    
    # Charger le modèle de base
    print("Chargement du modèle de base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Charger la configuration LoRA
    adapter_config = load_adapter_config()
    print(f"Configuration de l'adaptateur : {adapter_config}")
    
    # Charger le modèle avec l'adaptateur LoRA
    print("Chargement de l'adaptateur LoRA...")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        adapter_name="default"
    )
    
    # Fusionner l'adaptateur avec le modèle de base pour de meilleures performances
    print("Fusion de l'adaptateur avec le modèle...")
    model = model.merge_and_unload()
    
    print("Modèle chargé avec succès!")
    
except Exception as e:
    print(f"Erreur détaillée lors du chargement du modèle : {str(e)}")
    raise

class PredictionRequest(BaseModel):
    text: str
    max_length: int = 512
    temperature: float = 0.7
    num_return_sequences: int = 1

class PredictionResponse(BaseModel):
    generated_text: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Préparer le prompt dans le format TinyLlama Chat
        chat_prompt = f"<|system|>You are a helpful assistant.<|user|>{request.text}<|assistant|>"
        
        # Encoder le texte
        inputs = tokenizer(chat_prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(model.device)

        # Générer la réponse
        outputs = model.generate(
            **inputs,
            max_length=request.max_length,
            temperature=request.temperature,
            num_return_sequences=request.num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15
        )

        # Décoder la sortie et extraire uniquement la réponse de l'assistant
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Nettoyer la sortie pour ne garder que la réponse de l'assistant
        response_start = generated_text.find("<|assistant|>")
        if response_start != -1:
            generated_text = generated_text[response_start + len("<|assistant|>"):].strip()

        return PredictionResponse(generated_text=generated_text)

    except Exception as e:
        print(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)