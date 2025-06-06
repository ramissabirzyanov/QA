import os
import asyncio

from telegram.ext import ApplicationBuilder, MessageHandler, filters
from telegram.error import TelegramError
from telegram import Update
from telegram.ext import ContextTypes
import httpx
from dotenv import load_dotenv

from config.logger import logger


load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


async def ask_api(question: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/ask", json={"question": question})
        response.raise_for_status()
        data = response.json()
        return data["answer"]


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    chat_id = update.message.chat_id

    try:
        answer = await ask_api(user_question)
    except httpx.RequestError as e:
        logger.error(f"Error while asking API: {e}")
        answer = "Извините, произошла ошибка при обработке вашего запроса."
    try:
        await context.bot.send_message(chat_id=chat_id, text=answer)
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")


async def run_bot():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    await application.run_polling()
