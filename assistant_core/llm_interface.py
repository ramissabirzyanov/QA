import os
# import asyncio

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate

from config.settings import llm_settings, document_settings
from vector_storage.storage import VectorStorage
from config.logger import logger


PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    Ты — интеллектуальный помощник.
    Отвечай на вопросы на основе фактов из контекста.
    Не придумывай факты, которых нет контексте.
    Ты должен давать четкие, прямые ответы.
    Контекст:
    {context}
    Вопрос: {question}
    """
    )


def get_retriever() -> BaseRetriever:
    if not os.path.exists(f"{document_settings.FAISS_INDEX_PATH}"):
        logger.info("Хранилища нет — создаем...")
        storage = VectorStorage()
        storage.vectorise()

    embeddings = HuggingFaceEmbeddings(model_name=document_settings.MODEL_NAME)
    vectorstorage = FAISS.load_local(
        document_settings.FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstorage.as_retriever(search_kwargs={"k": 3})
    return retriever


def get_chain(prompt=PROMPT):
    llm = LlamaCpp(
        model_path=llm_settings.GGUF_MODEL,
        n_ctx=llm_settings.N_CTX,
        temperature=llm_settings.TEMPERATURE,
        max_tokens=llm_settings.MAX_TOKENS,
        verbose=llm_settings.VERBOSE,
        n_batch=llm_settings.N_BATCH,
        n_threads=llm_settings.N_THREADS
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain


# def get_answer(retriever: BaseRetriever, query: str) -> str:
#     chain = get_chain()
#     docs = retriever.invoke(query)
#     context = "\n\n".join([doc.page_content for doc in docs])
#     result = chain.invoke({"context": context, "question": query})
#     logger.info("\nНайденные документы:")
#     for doc in docs:
#         source = doc.metadata.get("source")
#         logger.info(f"[{source}] {doc.page_content[:50]}...")
#     return result


# async def get_answer_async(retriever, query: str) -> str:
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(None, get_answer, retriever, query)

async def get_answer_async(retriever: BaseRetriever, query: str) -> str:
    chain = get_chain()
    docs = await retriever.ainvoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    result = await chain.ainvoke({"context": context, "question": query})
    logger.info("\nНайденные документы:")
    for doc in docs:
        source = doc.metadata.get("source")
        logger.info(f"[{source}] {doc.page_content[:50]}...")
    return result
