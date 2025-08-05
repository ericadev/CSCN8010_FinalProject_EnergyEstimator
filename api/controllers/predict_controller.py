from fastapi import APIRouter, HTTPException
#from models.predict_request import PromptRequest
from api.models.predict_request import PromptRequest

#from services.predictor import predict_energy
from api.services.predictor import predict_energy



router = APIRouter()

@router.post("/predict")
def predict_energy_consumption(prompt: PromptRequest):
    try:
        predicted_value = predict_energy(prompt.prompt)
        if predicted_value is None:
            raise HTTPException(status_code=404, detail="Prediction could not be made.")
        return predicted_value
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    