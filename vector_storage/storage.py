import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from document.reader import get_reader
from qph.settings import settings
from qph.logger import logger


class VectorStorage:
    def __init__(self):
        self.data_dir = settings.DOCS_DIR
        self.model = settings.MODEL_NAME
        self.index_path = settings.FAISS_INDEX_PATH

    def vectorise(self) -> None:
        all_chunks = []
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            if not os.path.isfile(file_path):
                continue
            logger.info(f"Обрабатывается файл: {filename}")
            try:
                reader = get_reader(file_path)
                chunks = reader.get_chunks()
                all_chunks.extend(chunks)
            except ValueError as e:
                logger.info(f"Пропущен файл {filename}: {e}")
                continue
        logger.info(f"Количество чанков: {len(all_chunks)}")
        for i, doc in enumerate(all_chunks[:37:10]):
            filename = doc.metadata.get("source")
            logger.info(
                f"{i+1}. Файл: {filename}\n"
                f"Текст: {doc.page_content[:20]}..."
            )
        logger.info("Векторизация...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model
        )
        storage = FAISS.from_documents(all_chunks, embedding=embeddings)
        storage.save_local(self.index_path)
