import os

# import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_huggingface import HuggingFacePipeline
from langchain_ollama import OllamaLLM

from vector_storage.storage import VectorStorage
from qph.settings import settings
from qph.logger import logger
from langchain.prompts import PromptTemplate

def main():
    if not os.path.exists(f"{settings.FAISS_INDEX_PATH}"):
        logger.info("Хранилища нет — создаем...")
        storage = VectorStorage()
        storage.vectorise()

    PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        Ты — интеллектуальный помощник, который отвечает на вопросы, используя предоставленный контекст.
        Контекст:
        {context}
        Вопрос: {question}
        Ответ на только русском языке:
        """
        )

    embeddings = HuggingFaceEmbeddings(model_name=settings.MODEL_NAME)
    vectorstorage = FAISS.load_local(
        settings.FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = OllamaLLM(
        model="mistral",
        temperature=0.5,
        num_predict=256,
        verbose=True
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstorage.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    try:
        while True:
            query = input("\nВаш вопрос: ")
            if not query:
                break
            result = qa.invoke({"query": query})
            print("\nОтвет:", result["result"])
            logger.info("\nНайденные документы:")
            for doc in result["source_documents"]:
                source = doc.metadata.get("source")
                logger.info(f"[{source}] {doc.page_content[:100]}...")
    except KeyboardInterrupt:
        print("\nBYE", flush=True)


if __name__ == "__main__":
    main()
