import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

if __name__ == "__main__":
    load_dotenv()
    print("Retrieving...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=512)
    llm = ChatOpenAI()

    query = "What is vector database in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(f"Response not based on Vector DB: {result.content}")

    vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

    # https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combined_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combined_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(f"Response based on Vector DB: {result['answer']} with source {result['context']}")
