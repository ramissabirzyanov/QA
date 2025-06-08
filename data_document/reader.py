import os
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyMuPDFLoader
from config.settings import document_settings
from config.logger import logger


class BaseDocument(ABC):
    """Базовый класс для чтения и разбиения документа на чанки."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def read_document(self) -> list[Document]:
        pass

    def get_chunks(
        self,
        chunk_size=document_settings.CHUNK_SIZE,
        overlap=document_settings.CHUNK_OVERLAP
    ) -> list[Document]:
        docs = self.read_document()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        return splitter.split_documents(docs)


class DOCX_Reader(BaseDocument):
    def read_document(self) -> list[Document]:
        logger.info(f"Чтение DOCX: {self.file_path}")
        loader = Docx2txtLoader(self.file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = self.file_path
        return docs


class PDF_Reader(BaseDocument):
    def read_document(self) -> list[Document]:

        logger.info(f"Чтение PDF: {self.file_path}")
        loader = PyMuPDFLoader(self.file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = self.file_path
        return docs


def get_reader(file_path: str):
    """Возвращает подходящий ридер для файла по расширению."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".pdf":
        return PDF_Reader(file_path)
    elif ext == ".docx":
        return DOCX_Reader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
