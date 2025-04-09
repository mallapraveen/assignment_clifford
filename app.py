import os, json
import pickle
from typing import Dict, Optional
import chainlit as cl
import boto3
from chainlit.types import ThreadDict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.s3 import S3StorageClient
from chainlit.input_widget import Select, Switch, Slider
from src.rag.rag_pipeline import get_chat_chain_rerank, get_retriever_parent_child
from src.models.aws_models import (
    get_aws_chat_sonnet,
)

from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")


# storage_client = S3StorageClient(
#     bucket=os.getenv("BUCKET"),
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
# )


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///assignment.db")


@cl.cache
def get_chain_cache():
    retriever, bm25_retriever = get_retriever_parent_child("./data")
    chain = get_chat_chain_rerank(
        retriever, bm25_retriever, 0.5, get_aws_chat_sonnet(0, False)
    )
    return chain


@cl.on_chat_start
async def on_chat_start():
    chain = get_chain_cache()
    cl.user_session.set("chat_history", [])


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("chat_history", [])
    for message in thread["steps"]:
        if message["type"] == "user_message":
            cl.user_session.get("chat_history").append(
                {"role": "user", "content": message["output"]}
            )
        elif message["type"] == "assistant_message":
            cl.user_session.get("chat_history").append(
                {"role": "assistant", "content": message["output"]}
            )


@cl.on_message
async def main(message: cl.Message):
    chain = get_chain_cache()
    chat_history = cl.user_session.get("chat_history")
    cl_cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    res = await chain.acall(
        {"question": message.content, "chat_history": chat_history},
        callbacks=[cl_cb],
    )
    answer = res["answer"]
    source_documents = res["source_documents"]

    source_docs = []
    elements = []
    if source_documents:
        source_docs = list(
            set([source_doc.metadata["source"] for source_doc in source_documents])
        )
        for i in range(len(source_docs)):
            source_name = f"source_{i}"
            elements.append(
                cl.Pdf(
                    name=source_name,
                    display="inline",
                    path=source_docs[i],
                    page=1,
                )
            )

    chat_history.append((message.content, answer))

    await cl.Message(content=answer).send()  # , elements=elements


if __name__ == "__main__":
    retriever, bm25_retriever = get_retriever_parent_child("./data")
    qa_chain = get_chat_chain_rerank(
        retriever, bm25_retriever, 0.5, get_aws_chat_sonnet(0, False)
    )
    query = "What are the key similarities between GDPR and Brazil's LGPD?"
    value = qa_chain.invoke({"question": query, "chat_history": []})
    # qa_chain.acall()
    print(value["answer"])
