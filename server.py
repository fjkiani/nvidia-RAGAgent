from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import LongContextReorder  # Updated import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.base import RunnableSerializable
from langserve import add_routes
from fastapi import FastAPI
import uvicorn

# Initialize models
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct") | StrOutputParser()

# Load the index
docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)

# Create retriever chain
def docs2str(docs, title="Document"):
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

long_reorder = RunnableLambda(LongContextReorder().transform_documents)
retriever_chain = (
    RunnablePassthrough() | 
    docstore.as_retriever() | 
    long_reorder | 
    RunnableLambda(docs2str)
)

# Create generator chain
prompt = ChatPromptTemplate.from_template(
    "Answer the following question using only the provided context.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer: "
)

generator_chain = (
    prompt | 
    llm
)

# Create FastAPI app
app = FastAPI(title="RAG Assessment")

# Add routes
add_routes(
    app,
    retriever_chain,
    path="/retriever",
    input_type=str,
)

add_routes(
    app,
    generator_chain,
    path="/generator",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)