from pydantic import BaseModel
from typing import List

class LineItem(BaseModel):
    description: str
    amount: float

class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    invoice_date: str
    total_amount: float
    currency: str
    line_items: List[LineItem]