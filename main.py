import pandas as pd
from loguru import logger
from typing import Dict
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()
model = load('model/model.joblib')


class Input(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example='State-gov')
    fnlwgt: int = Field(..., example=77516)
    education: str = Field(..., example='Bachelors')
    education_num: int = Field(..., example=13, alias='education-num')
    marital_status: str = Field(..., example='Never-married',
                                alias='marital-status')
    occupation: str = Field(..., example='Adm-clerical')
    relationship: str = Field(..., example='Not-in-family')
    race: str = Field(..., example='White')
    sex: str = Field(..., example='Male')
    capital_gain: int = Field(..., example=2174, alias='capital-gain')
    capital_loss: int = Field(..., example=0, alias='capital-loss')
    hours_per_week: int = Field(..., example=40, alias='hours-per-week')
    native_country: str = Field(..., example='United-States',
                                alias='native-country')

    class Config:
        allow_population_by_field_name = True


@app.get("/")
async def welcome():
    return "Welcome to a demo ML web app!"


@app.post("/infer")
async def infer(input: Input) -> Dict[str, int]:
    input_df = pd.DataFrame([input.dict(by_alias=True)])
    logger.info(f"Received input: {input_df}")
    prediction = int(model.predict(input_df)[0])
    logger.info(f"Prediction: {prediction}")
    return {'prediction': prediction}
