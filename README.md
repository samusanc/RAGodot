# RAGodot Godot Documentation RAG

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline tailored for the Godot Engine documentation. It enables natural language querying of engine features, class references, and GDScript syntax.

## Technical Stack
* **Source Data:** Official Godot Engine Documentation.
* **Embedding Model:** Qwen3-8B.
* **Functionality:** Vector indexing and semantic retrieval for developer assistance.

## Implementation
The system processes documentation source files into chunks, generates high-dimensional embeddings using the Qwen3-8B model, and stores them in a vector database. This allows for accurate retrieval of engine-specific information that standard LLMs may lack or hallucinate.

## Usage
1. Ingest the documentation.
2. Initialize the vector store.
3. Query the assistant for Godot-related technical help.
