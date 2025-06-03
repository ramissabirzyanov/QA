import os
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyMuPDFLoader
from qph.settings import settings
from qph.logger import logger


class DocumentReaderBase(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunk_size = settings.CHUNK_SIZE
        self.overlap = settings.CHUNK_OVERLAP

    @abstractmethod
    def read_document(self) -> list[Document]:
        pass

    def get_chunks(self) -> list[Document]:
        docs = self.read_document()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap
        )
        return splitter.split_documents(docs)


class DOCX_Reader(DocumentReaderBase):
    def read_document(self) -> list[Document]:
        logger.info(f"Чтение DOCX: {self.file_path}")
        loader = Docx2txtLoader(self.file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = self.file_path
        return docs


class PDF_Reader(DocumentReaderBase):
    def read_document(self) -> list[Document]:

        logger.info(f"Чтение PDF: {self.file_path}")
        loader = PyMuPDFLoader(self.file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = self.file_path
        return docs

def get_reader(file_path: str):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".pdf":
        return PDF_Reader(file_path)
    elif ext == ".docx":
        return DOCX_Reader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

