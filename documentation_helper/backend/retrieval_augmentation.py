import os
from typing import Tuple, List

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def run_llm(query: str, chat_history: List[Tuple[str, str]]):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
    chat = ChatOpenAI(temperature=0)

    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke({"input": query, "chat_history": chat_history})
    transformed_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return transformed_result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain chain?")
    print(res['result'])