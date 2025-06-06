# import os
import asyncio

import uvicorn
from fastapi import FastAPI
from redis.asyncio import Redis
# from dotenv import load_dotenv

# from telegram import Bot
# from telegram.ext import ApplicationBuilder, MessageHandler, filters

from assistant_core.endpoints import router as api_router
from assistant_core.llm_interface import get_retriever, get_chain
from assistant_core.middlewares import RequestTimeMiddleware
from assistant_core.bot import run_bot
# from assistant_core.bot import handle_message
# from config.logger import logger


# load_dotenv()

# TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
# BASE_URL = os.getenv("BASE_URL")




async def lifespan(app: FastAPI):
    app.state.retriever = get_retriever()
    app.state.chain = get_chain()
    app.state.redis = Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    # telegram_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    # app.state.telegram_app = telegram_app

    # bot = Bot(token=TELEGRAM_TOKEN)
    # webhook_url = f"{BASE_URL}/telegram_webhook"
    # await bot.set_webhook(webhook_url)
    # logger.info(f"Webhook установлен: {webhook_url}")

    try:
        yield
    finally:
        # await bot.delete_webhook()
        await app.state.redis.close()
        await app.state.redis.connection_pool.disconnect()


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestTimeMiddleware)
app.include_router(api_router)


async def main():
    config = uvicorn.Config("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
    server = uvicorn.Server(config)

    # Запускаем одновременно FastAPI сервер и Telegram polling бота
    await asyncio.gather(
        server.serve(),
        run_bot(),
    )


if __name__ == "__main__":
    asyncio.run(main())