from langchain.storage import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from cachier import cachier


@cachier(cache_dir="./persist/cache")
def get_embeddings(
    model_name="Alibaba-NLP/gte-multilingual-base",
) -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_name="BAAI/bge-small-en-v1.5",
        # model_name="BAAI/bge-large-en-v1.5",
        # model_name="mixedbread-ai/mxbai-embed-large-v1",
        # model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
    )
    return embeddings


def get_cache_embeddings(
    model_name="Alibaba-NLP/gte-multilingual-base",
) -> CacheBackedEmbeddings:  # Alibaba-NLP/gte-multilingual-base, BAAI/llm-embedder
    embeddings = get_embeddings(model_name=model_name)
    store = LocalFileStore("./persist/indexes")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=model_name
    )
    return cached_embedder
