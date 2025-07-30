from typing import Set

import streamlit as st

from documentation_helper.backend.retrieval_augmentation import run_llm

st.header("LangChain Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your query here")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_sources(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    source_list = list(source_urls)
    source_list.sort()
    source_string = "Sources:\n"
    for i, source in enumerate(source_list):
        source_string += f"{i + 1}. {source}\n"
    return source_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])

        formatted_response = f"{generated_response['result']} \n\n Sources: {create_sources(sources)}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response['result']))

if st.session_state["chat_answer_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)
