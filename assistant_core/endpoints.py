from fastapi import APIRouter, Depends, Request
from redis.asyncio import Redis
from telegram import Update

from assistant_core.dependencies import (
    get_redis_dep,
    get_telegram_app_dep,
    get_assistant_dep
)
from assistant_core.schemas import QuerySchema, AnswerSchema
from assistant_core.assistant_interface import Assistant


router = APIRouter()


@router.post("/ask", response_model=AnswerSchema)
async def ask_question(
    query: QuerySchema,
    assistant: Assistant = Depends(get_assistant_dep),
    redis: Redis = Depends(get_redis_dep)
) -> AnswerSchema:
    """
    Обрабатывает POST-запрос с вопросом, возвращает ответ.
    """
    answer = await assistant.get_answer_async(query.question, redis)
    return AnswerSchema(answer=answer)


# @router.post("/telegram_webhook")
# async def telegram_webhook(request: Request, telegram_app=Depends(get_telegram_app_dep)):
#     json_update = await request.json()
#     update = Update.de_json(json_update, telegram_app.bot)
#     await telegram_app.process_update(update)
#     return {"status": "ok"}
