from  fastapi import FastAPI

import pickle
import numpy as np

from pydantic import BaseModel

with open("house_price_model.pkl","rb")as f:
    model=pickle.load(f)

app=FastAPI(title="House Price Predication API.")

class HouseData(BaseModel):

    size_sqft : float
    bedrooms:int

class Bulkdata(BaseModel):
    data :list[HouseData]


@app.get('/')
def home():
    return {"Message":"Welcome to the House Price Prediction API."}

@app.post('/predict')
def predict_price(data:HouseData):
     features=np.array([[data.size_sqft,data.bedrooms]])

     prediction=model.predict(features)
     return {"predicted_price":float(prediction[0])}

@app.post('/predict_batch')
def predict_batch(
    data:Bulkdata
):
    features=np.array([[house.size_sqft,house.bedrooms] for house in data.data])
    prediction=model.predict(features)
    return{"predicted_prices":prediction.tolist()}


