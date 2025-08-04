# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 2025 16:45

@author: Fatemeh Esmaeili
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# 2. Class which describes Bank Notes measurements
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

# 3. Create the app object
app = FastAPI()

# 4. Load the trained classifier
with open("bankNoteClassifier.pkl", "rb") as f:
    classifier = pickle.load(f)

# 5. Index route
@app.get("/")
async def index():
    return {"message": "Hello and Welcome to This FastAPI prototype!"}

# 6. Greeting route
@app.get("/{name}")
async def get_name(name: str):
    return {"message": f"I am happy to see you here, {name}"}

# 7. POST: Predict from banknote features
@app.post("/predict")
async def predict_banknote(data: BankNote):
    features = [[data.variance, data.skewness, data.curtosis, data.entropy]]
    prediction = classifier.predict(features)[0]

    result = "Fake note" if prediction > 0.5 else "It's a Bank note"
    return {"prediction": result}

# 8. PUT: Update prediction with new features
@app.put("/update_predict")
async def update_predict(data: BankNote):
    features = [[data.variance, data.skewness, data.curtosis, data.entropy]]
    prediction = classifier.predict(features)[0]

    result = "Fake note" if prediction > 0.5 else "It's a Bank note"
    return {
        "message": "Updated banknote features and prediction",
        "updated_features": data.dict(),
        "prediction": result
    }

# 9. Run the API
if __name__ == "__main__":
    uvicorn.run("asyncApp:app", host="127.0.0.1", port=9999, reload=True)


#uvicorn asyncApp:app --reload
