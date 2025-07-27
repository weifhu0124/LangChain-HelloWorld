from typing import Union, List

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, BaseTool
from langchain_openai import ChatOpenAI

from react_agent.callbacks import AgentCallBackHandler

load_dotenv()


@tool(description="Returns the length of a text by characters")
def get_text_length(text: str) -> int:
    """
    Returns the length of a text by characters
    """
    print(f"Running get_text_length for text: {text}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tool_list: List[BaseTool], name: str) -> BaseTool:
    for a_tool in tool_list:
        if a_tool.name == name:
            return a_tool
    raise ValueError(f"Tool with name {name} not found")


if __name__ == "__main__":
    tools = [get_text_length]

    # https://smith.langchain.com/hub/hwchase17/react
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    # observation is the result of the tool
    llm = ChatOpenAI(
        temperature=0,
        stop_sequences=["\nObservation", "Observation", "Observation:"],
        model="gpt-3.5-turbo",
        callbacks=[AgentCallBackHandler()]
    )

    intermediate_step = []

    agent = (
            {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])}
             | prompt
             | llm
             | ReActSingleInputOutputParser()
    )

    while True:
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the text 'Hello, world!'?",
                "agent_scratchpad": intermediate_step
            }
        )
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            intermediate_step.append((agent_step, str(observation)))

        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the text 'Hello, world!'?",
                "agent_scratchpad": intermediate_step
            }
        )

        if isinstance(agent_step, AgentFinish):
            print(agent_step.return_values)
            break
