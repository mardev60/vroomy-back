FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python une à une
RUN pip install --no-cache-dir fastapi==0.104.1
RUN pip install --no-cache-dir pydantic==2.5.2
RUN pip install --no-cache-dir torch==2.1.1
RUN pip install --no-cache-dir transformers==4.35.2
RUN pip install --no-cache-dir peft==0.7.1
RUN pip install --no-cache-dir uvicorn==0.24.0

# Copie des fichiers du projet
COPY app.py .
COPY api.py .
COPY car_price_model_final/ ./car_price_model_final/

# Exposition du port
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 