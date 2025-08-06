from fastapi import FastAPI
#from controllers import predict_controller
from api.controllers import predict_controller

app = FastAPI(title="Energy Consumption Predictor")

app.include_router(predict_controller.router, prefix="/api/predictor", tags=["Prediction"])

# to run the FastAPI app, use the command:
# uvicorn main:app --reload


