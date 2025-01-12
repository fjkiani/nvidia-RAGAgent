from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

def test_embeddings():
    # List available models
    print("Available embedding models:")
    available_models = [m for m in NVIDIAEmbeddings.get_available_models() if "embed" in m.id]
    for model in available_models:
        print(f"- {model.id}")

    # Initialize the embedding model
    print("\nInitializing embedding model...")
    embedder = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-mistral-7b-v2",
        truncate="END"
    )

    # Test some sample texts
    texts = [
        "RAG models combine retrieval with generation",
        "Large language models can process text",
        "Neural networks learn patterns in data"
    ]

    print("\nGenerating embeddings for test texts...")
    embeddings = embedder.embed_documents(texts)
    
    print(f"\nEmbedding shape: {len(embeddings)}x{len(embeddings[0])}")
    
    # Calculate similarities between embeddings
    print("\nCalculating similarities between texts:")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            emb1 = np.array(embeddings[i])
            emb2 = np.array(embeddings[j])
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"Similarity between '{texts[i]}' and '{texts[j]}': {similarity:.3f}")

    # Test creating a small FAISS index
    print("\nTesting FAISS index creation...")
    index = FAISS.from_texts(texts, embedder)
    
    # Test querying
    query = "What is retrieval-augmented generation?"
    print(f"\nTesting query: '{query}'")
    results = index.similarity_search_with_score(query, k=2)
    
    print("\nResults:")
    for doc, score in results:
        print(f"Score: {score:.3f}, Text: {doc.page_content}")

if __name__ == "__main__":
    test_embeddings() 