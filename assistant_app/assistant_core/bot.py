import os

from telegram.error import TelegramError
from telegram import Update
from telegram.ext import ContextTypes, Application
import httpx
from dotenv import load_dotenv

from assistant_app.config.logger import logger


load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


async def ask_api(question: str) -> str:
    """Отправляет вопрос в backend API и возвращает ответ."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        response = await client.post(f"{BASE_URL}/ask", json={"question": question})
        response.raise_for_status()
        data = response.json()
        return data["answer"]


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает входящие текстовые сообщения от пользователя."""
    user_question = update.message.text
    chat_id = update.message.chat_id
    try:
        answer = await ask_api(user_question)
    except httpx.RequestError as e:
        logger.error(f"Error while asking API: {e}")
        logger.exception("Error while asking API")
        answer = "Извините, произошла ошибка при обработке вашего запроса:."
    try:
        await context.bot.send_message(chat_id=chat_id, text=answer)
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет приветственное сообщение при команде /start."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(
            "Привет! Я очень постараюсь ответить на твои вопросы о компании. "
            "Давай начнём."
        )
    )


async def run_polling(telegram_app: Application):
    """Запускает бота в режиме polling"""
    await telegram_app.start()
    await telegram_app.updater.start_polling()
