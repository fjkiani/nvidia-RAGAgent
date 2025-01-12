from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from typing import Dict, Any, List, Union
import os

# Print available embedding models
print("Available embedding models:")
available_models = [m for m in NVIDIAEmbeddings.get_available_models() if "embed" in m.id]
for model in available_models:
    print(f"- {model.id}")

# Initialize embedding model
print("\nInitializing embedding model...")
embedder = NVIDIAEmbeddings(model="nv-embedqa-mistral-7b-v2", model_type=None)

# Initialize LLM
llm = ChatOpenAI()

# Load or create the FAISS index
print("Loading FAISS index...")
try:
    docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully!")
    
    # Test the index
    test_query = "What are RAG models?"
    test_results = docstore.similarity_search(test_query, k=1)
    if test_results:
        print("Index test successful - found matching documents")
        print(f"Sample result: {test_results[0].page_content[:200]}...")
    else:
        print("Warning: Index test returned no results")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    raise  # Let the error propagate since we need a working index

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="RAG-enabled API server using LangChain's Runnable interfaces"
)

# Basic chat endpoint
add_routes(
    app,
    llm | StrOutputParser(),
    path="/basic_chat",
)

# Retriever endpoint - return raw documents for frontend processing
def extract_query(input_data: Any) -> str:
    if isinstance(input_data, str):
        return input_data
    elif isinstance(input_data, dict):
        if "input" in input_data:
            inner_input = input_data["input"]
            if isinstance(inner_input, dict):
                return inner_input.get("question", "")
            return str(inner_input)
        return input_data.get("question", "")
    return str(input_data)

def retrieve_docs(query: Any) -> List:
    try:
        search_query = extract_query(query)
        print(f"Processing query: {search_query}")
        
        if not search_query:
            print("Empty query received")
            return []
            
        docs = docstore.similarity_search(search_query, k=4)
        print(f"Found {len(docs)} documents")
        return docs
    except Exception as e:
        print(f"Error in retrieve_docs: {e}")
        return []

# Create a simple retriever chain
retriever_chain = RunnableLambda(retrieve_docs)

add_routes(
    app,
    retriever_chain,
    path="/retriever",
)

# Generator endpoint
def format_docs(docs: List) -> str:
    try:
        formatted_docs = []
        for doc in docs:
            title = getattr(doc.metadata, 'title', 'Document')
            content = doc.page_content.strip()
            formatted_docs.append(f"[From paper '{title}']\n{content}")
        return "\n\n".join(formatted_docs)
    except Exception as e:
        print(f"Error in format_docs: {e}")
        return ""

rag_prompt = ChatPromptTemplate.from_template(
    "You are a knowledgeable AI assistant tasked with answering questions based on academic papers and research. "
    "You have been provided with relevant excerpts from research papers as context. "
    "You must ONLY use information from this context to answer the question - do not use any external knowledge. "
    "If the context contains relevant information, even if partial, use it to provide the best possible answer. "
    "When answering:\n"
    "1. Start with the most relevant information from the context\n"
    "2. Quote specific passages when possible\n"
    "3. Cite the papers you're drawing information from\n"
    "4. If the context only partially answers the question, provide what you can and explain what's missing\n"
    "5. If you truly find no relevant information in the context, explain that clearly\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer: "
)

def process_input(input_dict: Any) -> Dict[str, str]:
    query = extract_query(input_dict)
    return {"question": query}

# Build the RAG chain with improved error handling
rag_chain = (
    RunnablePassthrough()
    | RunnableLambda(process_input)
    | {
        "question": lambda x: x["question"],
        "context": RunnableLambda(lambda x: format_docs(retrieve_docs(x["question"])))
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

add_routes(
    app,
    rag_chain,
    path="/generator",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
