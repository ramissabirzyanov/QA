import os

from redis.asyncio import Redis
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_community.llms.vllm import VLLM
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


def get_llm(llm_type: str) -> Runnable:
    """Фабрика для выбора LLM: 'llamacpp' или 'vllm'"""
    if llm_type == "vllm":  # набросок
        llm = VLLM(
            model="yandex/YandexGPT-5-Lite-8B-instruct",
            max_new_tokens=llm_settings.MAX_TOKENS,
            temperature=llm_settings.TEMPERATURE,
            top_p=0.9,
            model_kwargs={"gpu_memory_utilization": 0.9},
            verbose=llm_settings.VERBOSE
        )
    elif llm_type == "llamacpp":
        llm = LlamaCpp(
            model_path=llm_settings.GGUF_MODEL,
            n_ctx=llm_settings.N_CTX,
            temperature=llm_settings.TEMPERATURE,
            max_tokens=llm_settings.MAX_TOKENS,
            verbose=llm_settings.VERBOSE,
            n_batch=llm_settings.N_BATCH,
            n_threads=llm_settings.N_THREADS,
            n_gpu_layers=llm_settings.N_GPU_LAYERS
        )
    else:
        raise ValueError(f"Неизвестный тип llm: {llm_type}")
    return llm


def get_chain(llm_type: str = "llamacpp", prompt: PromptTemplate = PROMPT) -> Runnable:
    llm = get_llm(llm_type)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain


async def get_answer_async(
    chain: Runnable,
    retriever: BaseRetriever,
    query: str, redis: Redis
) -> str:
    cache_key = f"qa_cache:{query}"

    cached_answer = await redis.get(cache_key)
    if cached_answer:
        return cached_answer

    docs = await retriever.ainvoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    result = await chain.ainvoke({"context": context, "question": query})
    logger.info("\nНайденные документы:")
    for doc in docs:
        source = doc.metadata.get("source")
        logger.info(f"[{source}] {doc.page_content[:50]}...")
    await redis.set(cache_key, result)
    return result
