from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter

if __name__ == "__main__":
    load_dotenv()

    pdf_loader = PyPDFLoader("/Users/weifenghu/Desktop/Weifeng/Study/Machine-Learning/LangChain/vector_db/pdf/react.pdf")
    documents = pdf_loader.load()
    print(f"Loaded {len(documents)} documents")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(docs)} chunks")

    embeddings = OpenAIEmbeddings()
    # in-memory local vector db
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    print("Vector store created")

    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combined_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_prompt
    )
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(), combined_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
