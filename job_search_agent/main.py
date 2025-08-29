from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from job_search_agent.prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from job_search_agent.schemas import AgentResponse

load_dotenv()

"""
Tool is a wrapper of Python function that contains name, description, argument and other metadata
"""
tools = [TavilySearch()]
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instruction = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"]
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(llm, tools, react_prompt_with_format_instruction)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    result = agent_executor.invoke(input={
        "input": "search for 3 job postings for an Senior Software Engineer using LLM/RAG in Seattle or remote on linkedin and list their details"
    })
    print(result)


if __name__ == "__main__":
    main()
