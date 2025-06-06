from fastapi import Request


def get_retriever_dep(request: Request):
    return request.app.state.retriever


def get_chain_dep(request: Request):
    return request.app.state.chain
