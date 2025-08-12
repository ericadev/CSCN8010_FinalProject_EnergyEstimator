from fastapi import APIRouter, HTTPException
from api.models.predict_request import PromptRequest
from api.services.predictor import predict_energy, predict_energy_by_verbs

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


@router.post("/predict_by_verbs")
def predict_energy_consumption_by_verbs(prompt: PromptRequest):
    try:
        predicted_value = predict_energy_by_verbs(prompt.prompt)
        if predicted_value is None:
            raise HTTPException(status_code=404, detail="Prediction could not be made.")
        return predicted_value
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))