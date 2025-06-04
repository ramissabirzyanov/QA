import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from config.logger import logger


class RequestTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time)
        endpoint = request.url.path
        logger.info(f"Запрос к {endpoint} выполнен за {process_time:.2f} с")
        return response
