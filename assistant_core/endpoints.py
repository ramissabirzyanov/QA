from fastapi import APIRouter


from schemas import QuerySchema, AnswerSchema
from assistant_core.llm_interface import get_answer_async


router = APIRouter()


@router.post("/ask", response_model=AnswerSchema)
async def ask_question(query: QuerySchema) -> AnswerSchema:
    answer = await get_answer_async(query.question)
    return AnswerSchema(answer=answer)
