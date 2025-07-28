import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

if __name__ == '__main__':
    load_dotenv()
    print("Loading...")
    loader = TextLoader("/Users/weifenghu/Desktop/Weifeng/Study/Machine-Learning/LangChain/vector_db/mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    # split smaller enough to fit in context window
    # but larger enough so that human can understand the context
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    print("Embedding...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=512)

    print("Ingesting...")
    # commented out to prevent duplicate upload
    # PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
