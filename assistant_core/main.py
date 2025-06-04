import uvicorn
from fastapi import FastAPI

from assistant_core.endpoints import router as api_router
from assistant_core.llm_interface import get_chain
from assistant_core.llm_interface import get_retriever


async def lifespan(app: FastAPI):
    app.state.chain = get_chain()
    app.state.retriever = get_retriever()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
