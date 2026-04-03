import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# 1. Set up the client
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Define the Qwen3 embedding function (assuming Ollama is running)
qwen_ef = OllamaEmbeddingFunction(
    model_name="qwen3-embedding:8b",
    url="http://192.168.0.36:11434/api/embeddings"
)

# 3. Create or get collection using that specific function
collection = client.get_or_create_collection(
    name="godot_docs", 
    embedding_function=qwen_ef
)


collection.add(
    ids=["id1", "id2"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ]
)

results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print(results)
