import os
import asyncio

import uvicorn
from fastapi import FastAPI
from redis.asyncio import Redis
from dotenv import load_dotenv

from telegram.ext import ApplicationBuilder, MessageHandler, filters, CommandHandler

from assistant_app.assistant_core.endpoints import router as api_router
from assistant_app.assistant_core.middlewares import RequestTimeMiddleware
from assistant_app.assistant_core.bot import handle_message, run_polling, start_command
from assistant_app.assistant_core.assistant import Assistant
from assistant_app.config.logger import logger


load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
BASE_URL = os.getenv("BASE_URL")


async def lifespan(app: FastAPI):
    app.state.assistant = Assistant()
    app.state.redis = Redis(host='localhost', port=6379, db=0, decode_responses=True)

    telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    telegram_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    telegram_app.add_handler(CommandHandler("start", start_command))

    app.state.telegram_app = telegram_app

    bot = telegram_app.bot

    # webhook_url = f"{BASE_URL}/telegram_webhook"
    # await bot.set_webhook(webhook_url)
    # logger.info(f"Webhook установлен: {webhook_url}")

    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook удалён. Переход на polling.")

    asyncio.create_task(run_polling(telegram_app))
    logger.info("Бот запущен в режиме polling")

    try:
        yield
    finally:
        # await bot.delete_webhook()
        await app.state.redis.close()
        await app.state.redis.connection_pool.disconnect()


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestTimeMiddleware)
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
