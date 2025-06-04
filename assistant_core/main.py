import os

from vector_storage.storage import VectorStorage
from config.settings import document_settings
from config.logger import logger

from assistant_core.llm_interface import get_retriever, get_chain


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
