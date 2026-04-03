import ollama

# Initialize the client (defaults to localhost:11434)
client = ollama.Client(host='http://192.168.0.36:11434')

response = client.embed(
    model='qwen3-embedding:8b',
    input='Testing the Qwen3 embedding model.'
)

# Print the first 5 dimensions of the vector to verify
print(f"Embedding dimensions: {len(response['embeddings'][0])}")
print(f"First 5 values: {response['embeddings'][0][:5]}")
