from fastapi import APIRouter, Depends, Request
from redis.asyncio import Redis
from telegram import Update

from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from assistant_core.dependencies import (
    get_retriever_dep,
    get_chain_dep,
    get_redis_dep,
    get_telegram_app_dep
)
from assistant_core.schemas import QuerySchema, AnswerSchema
from assistant_core.llm_interface import get_answer_async


router = APIRouter()


@router.post("/ask", response_model=AnswerSchema)
async def ask_question(
    query: QuerySchema,
    chain: Runnable = Depends(get_chain_dep),
    retriever: BaseRetriever = Depends(get_retriever_dep),
    redis: Redis = Depends(get_redis_dep)
) -> AnswerSchema:
    answer = await get_answer_async(chain, retriever, query.question, redis)
    return AnswerSchema(answer=answer)


# @router.post("/telegram_webhook")
# async def telegram_webhook(request: Request, telegram_app = Depends(get_telegram_app_dep)):
#     json_update = await request.json()
#     update = Update.de_json(json_update, telegram_app.bot)
#     await telegram_app.process_update(update)
#     return {"status": "ok"}
