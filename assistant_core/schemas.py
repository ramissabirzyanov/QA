from pydantic import BaseModel


class QuerySchema(BaseModel):
    question: str

class AnswerSchema(BaseModel):
    answer: str