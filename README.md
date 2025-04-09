Step to setup locally:

1. python -m venv ./env
2. source ./env/bin/activate
3. pip install -r requirements.txt


Experimentations:

rag_pipeline.ipynb

1. Tried using semantic, keyword, ensemble and ensemble with re ranking search. Found that later performed well in most cases.
2. Simple Q&A can be done on the notebook.
3. For chat capability, use the chainlit app. `chainlit run app.py`

Inside Folder Notebooks:
1. database.ipynb - create the database for data persistance needed for chainlit app.
2. models.ipynb - used for testing aws models
3. retrieval-evaluation.ipynb - used llama-index for generating retrieval evaluation data and evaluation was done below rag pipeline:
    1. semantic search
    2. Hybrid search
    3. Hybrid search with reranker
    used these metrics to track - "precision", "recall", "hit_rate", "ap", "mrr", "ndcg"
4. synthetic-dataset-generation-evaluation.ipynb - Used this notebook for generating synthetic data for generation evaluation.
5. generation-evaluation.ipynb - LLM based evaluation and nlp metrics eval.
    nlp based - rouge, blue, token_overlap, semantic similarity
    llm based - LLMContextPrecisionWithReference, LLMContextRecall, ContextEntityRecall, ResponseRelevancy, Faithfulness, FactualCorrectness