from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from inference import predict


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class PredictResponse(BaseModel):
    prediction: str


app = FastAPI()


@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    prediction = predict(request.text)
    return PredictResponse(prediction=prediction)
