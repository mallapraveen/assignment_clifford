import datetime
from pathlib import Path
from typing import Tuple, Union
from cachier import cachier

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
)
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader,
)
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.models.embeddings import get_cache_embeddings
from src.prompts import condense_question_template, qa_template

from langchain.memory import ConversationBufferMemory


@cachier(cache_dir="./persist/cache")
def load_documents(
    file_path: str,
) -> Union[TextLoader, Docx2txtLoader, DirectoryLoader]:
    file_suffix = Path(file_path).suffix
    if file_suffix == ".txt":
        return TextLoader(file_path).load()
    elif file_suffix == ".docx":
        return Docx2txtLoader(file_path).load()
    else:
        documents = DirectoryLoader(
            file_path,
            glob="*.pdf",
            # loader_kwargs={
            #     "strategy": "hi_res",
            # },
        ).load()
        return documents


def get_retriever_parent_child(
    file_path: str, model_name: str = "BAAI/llm-embedder"
) -> Tuple[ParentDocumentRetriever, BM25Retriever]:
    documents = load_documents(file_path)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
    )

    embeddings = get_cache_embeddings(model_name)
    vector_store_local = f"./persist/chroma/{model_name}"
    try:
        # vectorstore = FAISS.load_local(
        #     vector_store_local,
        #     embeddings,
        #     # allow_dangerous_deserialization=True,
        # )
        vectorstore = Chroma(
            persist_directory=vector_store_local, embedding_function=embeddings
        )
    except:
        vectorstore = Chroma.from_documents(
            documents, embeddings, persist_directory=vector_store_local
        )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3},
    )
    retriever.add_documents(documents)

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    return retriever, bm25_retriever


def get_chat_chain_rerank(
    retriever: ParentDocumentRetriever,
    bm25_retriever: BM25Retriever,
    weights: float,
    llm,
    prompt: str = "",
) -> RetrievalQA:

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=[1 - weights, weights],
        search_kwargs={"k": 3},
    )

    model = HuggingFaceCrossEncoder(
        model_name="mixedbread-ai/mxbai-rerank-base-v1"
    )  # mixedbread-ai/mxbai-rerank-base-v1,BAAI/bge-reranker-base
    compressor = CrossEncoderReranker(model=model, top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
        search_kwargs={"k": 3},
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    condense_question_prompt = ChatPromptTemplate.from_template(
        condense_question_template
    )
    qa_prompt = ChatPromptTemplate.from_template(qa_template)
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        compression_retriever,
        # condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={
            "prompt": qa_prompt,
        },
        return_source_documents=True,
        memory=memory,
        verbose=False,
    )

    return chain


def get_qa_chain_rerank(
    retriever: ParentDocumentRetriever,
    bm25_retriever: BM25Retriever,
    weights: float,
    llm,
    prompt: str = "",
) -> RetrievalQA:

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=[1 - weights, weights],
        search_kwargs={"k": 3},
    )

    model = HuggingFaceCrossEncoder(
        model_name="mixedbread-ai/mxbai-rerank-base-v1"
    )  # mixedbread-ai/mxbai-rerank-base-v1,BAAI/bge-reranker-base
    compressor = CrossEncoderReranker(model=model, top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
        search_kwargs={"k": 3},
    )

    if prompt == "":
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=compression_retriever,
            return_source_documents=True,
        )
    else:
        prompt = PromptTemplate(
            template=prompt, input_variables=["context", "question"]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    return qa_chain
