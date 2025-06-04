from fastapi import APIRouter


from schemas import QuerySchema, AnswerSchema
from assistant_core.llm_interface import get_answer


router = APIRouter()


@router.post("/ask", response_model=AnswerSchema)
def ask_question(query: QuerySchema) -> AnswerSchema:
    answer = get_answer(query.question)
    return AnswerSchema(answer=answer)
