import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate

from vector_storage.storage import VectorStorage
from config.settings import document_settings, llm_settings
from config.logger import logger


PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    Ты — интеллектуальный помощник. Используй информацию из контекста, чтобы ответить на вопрос.
    Инструкции:
    - Дай развернутый ответ, если конетекст позволяет.
    - Не добавляй информацию, если её нет в контексте.
    - Если данных недостаточно — скажи: "У меня недостаточно информации, чтобы ответить."
    - Ответ должен быть на русском языке.
    Контекст:
    {context}
    Вопрос: {question}
    """
    )


def get_retriever():
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
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain


def main():
    if not os.path.exists(f"{document_settings.FAISS_INDEX_PATH}"):
        logger.info("Хранилища нет — создаем...")
        storage = VectorStorage()
        storage.vectorise()

    retriever = get_retriever()
    chain = get_chain()
    try:
        while True:
            query = input("\nВаш вопрос: ")
            if not query:
                break
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            result = chain.invoke({"context": context, "question": query})
            print("\nОтвет:", result)
            logger.info("\nНайденные документы:")
            for doc in docs:
                source = doc.metadata.get("source")
                logger.info(f"[{source}] {doc.page_content[:50]}...")
    except KeyboardInterrupt:
        print("\nBYE", flush=True)


if __name__ == "__main__":
    main()
