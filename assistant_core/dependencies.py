from fastapi import Request


def get_chain(request: Request):
    return request.app.state.chain


def get_retriever(request: Request):
    return request.app.state.retriever
