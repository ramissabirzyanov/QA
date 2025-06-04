import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.llamacpp import LlamaCpp
from vector_storage.storage import VectorStorage
from qph.settings import settings
from qph.logger import logger
from langchain.prompts import PromptTemplate


def main():
    if not os.path.exists(f"{settings.FAISS_INDEX_PATH}"):
        logger.info("Хранилища нет — создаем...")
        storage = VectorStorage()
        storage.vectorise()

    embeddings = HuggingFaceEmbeddings(model_name=settings.MODEL_NAME)
    vectorstorage = FAISS.load_local(
        settings.FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    Ты — интеллектуальный помощник. Используй информацию из контекста, чтобы ответить на вопрос.
    Инструкции:
    - Не добавляй информацию, если её нет в контексте.
    - Если данных недостаточно — скажи: "В предоставленном контексте недостаточно информации, чтобы ответить."
    - Ответ должен быть на русском языке.
    Контекст:
    {context}
    Вопрос: {question}
    """
    )

    llm = LlamaCpp(
        model_path="gguf_models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        n_ctx=2048,
        temperature=0.5,
        verbose=False,
        n_batch=128,
        max_tokens=512
    )
    retriever = vectorstorage.as_retriever(search_kwargs={"k": 3})
    parser = StrOutputParser()
    chain = PROMPT | llm | parser
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
