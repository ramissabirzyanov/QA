from fastapi import APIRouter, Depends

from assistant_core.dependencies import get_retriever
from assistant_core.schemas import QuerySchema, AnswerSchema
from assistant_core.llm_interface import get_answer_async


router = APIRouter()


@router.post("/ask", response_model=AnswerSchema)
async def ask_question(
    query: QuerySchema,
    retriever=Depends(get_retriever)
) -> AnswerSchema:
    answer = await get_answer_async(retriever, query.question)
    return AnswerSchema(answer=answer)
