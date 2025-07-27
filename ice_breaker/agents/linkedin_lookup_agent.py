import os

from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_tavily import TavilySearch


def lookup(name: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
    )
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL starting with https"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    tools_for_agent = [
        # Initialize Tavily Search Tool
        TavilySearch(
            max_results=5,
            topic="general",
            name="Search Tavily for LinkedIn Profile"
        )
    ]

    # https://smith.langchain.com/hub/hwchase17/react
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url


if __name__ == "__main__":
    load_dotenv()
    lookup("Weifeng Hu AWS")
