from fastapi import Request


async def get_redis_dep(request: Request):
    return request.app.state.redis

async def get_chain_dep(request: Request):
    return request.app.state.chain

async def get_retriever_dep(request: Request):
    return request.app.state.retriever