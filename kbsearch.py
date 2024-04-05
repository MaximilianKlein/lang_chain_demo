from langchain_community.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Create Embeddings and Initialize Chroma
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
results = vectorstore.similarity_search("How do you sign up for portuguese health insurance?")

# Displaying results
for result in results:
    print(f"Document: {result.page_content[:100]}, Metadata: {result.metadata}")
