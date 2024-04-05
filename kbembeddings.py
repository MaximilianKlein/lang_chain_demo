import json
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Load the JSON file
with open('kb/kb.json', 'r') as file:
    data = json.load(file)

# Extract Document Information and filter for published documents
docs = []
for item in data:
    document = json.loads(item['val'])
    if document.get('state') == "PUBLISHED":
        # Combine title and content for the document's page_content
        page_content = f"{document.get('title', '')}\n\n{document.get('content', '')}"
        # Prepare metadata with documentID
        metadata = {"documentID": document.get('documentID'), "title": document.get("title")}
        # Create a Document object and append to the docs list
        docs.append(Document(page_content=page_content, metadata=metadata))

# Create Embeddings and Initialize Chroma
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# Note: Assuming Chroma accepts a list of Document objects directly. If not, adjust accordingly.
vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

# Persist the vector store
vectorstore.persist()
