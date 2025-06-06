import uvicorn
from fastapi import FastAPI
from redis.asyncio import Redis

from assistant_core.endpoints import router as api_router
from assistant_core.llm_interface import get_retriever, get_chain
from assistant_core.middlewares import RequestTimeMiddleware


async def lifespan(app: FastAPI):
    app.state.retriever = get_retriever()
    app.state.chain = get_chain()
    app.state.redis = Redis(host='localhost', port=6379, db=0, decode_responses=True)
    try:
        yield
    finally:
        await app.state.redis.close()
        await app.state.redis.connection_pool.disconnect()


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestTimeMiddleware)
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
