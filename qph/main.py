import os

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

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

    model_id = "ai-forever/rugpt3medium_based_on_gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=settings.HF_API_TOKEN, cache_dir="./model_cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        token=settings.HF_API_TOKEN,
        cache_dir="./model_cache"
    )
    logger.info("model и tokenizer загружены")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        return_full_text=False,
        temperature=0.7
    )

    llm = HuggingFacePipeline(pipeline=pipe)

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
                logger.info(f"[{source}] {doc.page_content[:100]}")
    except KeyboardInterrupt:
        print("\nBYE", flush=True)

if __name__ == "__main__":
    main()

