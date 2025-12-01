from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices in India",
    version="1.0.0",
)

# Setup templates directory
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = "house_price_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None

class HouseFeatures(BaseModel):
    size_sqft: float
    bedrooms: int
    city: str
    location: str

# Sample city data with areas
CITY_AREAS = {
    "Mumbai": ["Bandra", "Andheri", "Dadar", "Parel", "Worli", "Juhu", "Powai"],
    "Delhi": ["Janakpuri", "Rohini", "Connaught Place", "Saket", "Dwarka"],
    "Bangalore": ["Indiranagar", "Koramangala", "Whitefield", "MG Road", "Electronic City"],
    "Hyderabad": ["Gachibowli", "Hitech City", "Jubilee Hills", "Banjara Hills"],
    "Pune": ["Koregaon Park", "Hinjewadi", "Wakad", "Baner", "Kothrud"],
    "Chennai": ["Anna Nagar", "T. Nagar", "OMR", "Velachery", "Adyar"]
}

@app.get("/api/areas/{city}")
async def get_areas(city: str):
    """Get areas for a specific city"""
    city = city.capitalize()
    if city not in CITY_AREAS:
        raise HTTPException(status_code=404, detail="City not found")
    return {"areas": CITY_AREAS[city]}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "House Price Predictor",
            "cities": list(CITY_AREAS.keys()),
            "city_areas": CITY_AREAS
        }
    )

@app.post("/predict")
async def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_data = {
            'size_sqft': [features.size_sqft],
            'bedrooms': [features.bedrooms],
            'city': [features.city],
            'location': [features.location],
        }
        
        prediction = model.predict(pd.DataFrame(input_data))[0]
        
        return {
            "predicted_price": round(float(prediction), 2),
            "currency": "INR",
            "features": features.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
