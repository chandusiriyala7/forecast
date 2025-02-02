from pydantic import BaseModel

class QueryDate(BaseModel):
    date: str  # Date in "YYYY-MM-DD" format