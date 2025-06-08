from fastapi import Request


async def get_redis_dep(request: Request):
    return request.app.state.redis


async def get_telegram_app_dep(request: Request):
    return request.app.state.telegram_app


def get_assistant_dep(request: Request):
    return request.app.state.assistant
