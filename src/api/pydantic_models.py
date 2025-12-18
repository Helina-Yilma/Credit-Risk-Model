from pydantic import BaseModel

class TransactionInput(BaseModel):
    Amount: float
    Value: float
    TransactionHour: int
    TransactionDay: int
    Average_Amount: float
    Transaction_Count: int

class PredictionOutput(BaseModel):
    risk_probability: float
    is_high_risk: int