import os
from typing import List, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def custom_retrieval_prompt() -> PromptTemplate:
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum to keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""

    return PromptTemplate.from_template(template)


def format_docs(docs: List[Document])-> str:
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    load_dotenv()
    custom_rag_prompt = custom_retrieval_prompt()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=512)
    llm = ChatOpenAI()
    query = "What is vector database in machine learning?"

    vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

    rag_chain = (
        {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)
