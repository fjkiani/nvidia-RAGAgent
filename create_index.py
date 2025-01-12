from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ArxivLoader
import os

# Initialize embeddings
embedder = NVIDIAEmbeddings(model='nvidia/nv-embed-v1', truncate='END')

# Load documents
docs = [
    ArxivLoader(query='2306.05685').load(),  # LLM-as-a-Judge (recent paper)
    ArxivLoader(query='2005.11401').load(),  # RAG Paper
    ArxivLoader(query='2205.00445').load(),  # MRKL Paper
]

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    separators=['\n\n', '\n', '.', ';', ',', ' ', ''],
)

# Process documents
all_splits = []
for doc in docs:
    content = doc[0].page_content
    if 'References' in content:
        doc[0].page_content = content[:content.index('References')]
    splits = text_splitter.split_documents(doc)
    splits = [c for c in splits if len(c.page_content) > 200]
    all_splits.extend(splits)

# Create and save FAISS index
docstore = FAISS.from_documents(all_splits, embedder)
docstore.save_local('docstore_index')

# Compress the index
os.system('tar czvf docstore_index.tgz docstore_index')
print('Index created and saved successfully!') 