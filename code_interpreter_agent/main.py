from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
import qrcode

load_dotenv()


def main():
    print("Starting...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    tools = [PythonREPLTool()] # DANGER - do not use in production
    python_agent = create_react_agent(
        prompt=prompt,
        tools=tools,
        llm=llm
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    # python_agent_executor.invoke(input={
    #     "input": """Generate and save in a folder called qrcode under current working direction 5 QRCodes that
    #     point to https://www.linkedin.com/in/weifeng-hu-7b7364105/. If the folder does not exist, create it"""
    # })

    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=llm,
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    # csv_agent.invoke(input={
    #     "input": """Which writer wrote the most episodes and how many did they write?"""
    # })

    router_tool = [
        Tool(
            name="Python agent",
            func=python_agent_executor.invoke,
            description="""Useful when you need to transform natural language to pythin and execute the python code,
            returning the results of the code execution
            DOES NOT ACCEPT CODE AS INPUT
            """
        ),
        Tool(
            name="CSV agent",
            func=csv_agent_executor.invoke,
            description="""Useful when you need to answer question over episode_info.csv file,
            takes an inut the entire question and return the answer after running pandas calculations.
            """
        ),
        PythonREPLTool()
    ]

    router_prompt = base_prompt.partial(instructions="")
    router_agent = create_react_agent(
        llm=llm,
        prompt=router_prompt,
        tools=router_tool
    )
    router_agent_executor = AgentExecutor(agent=router_agent, tools=router_tool, verbose=True)
    router_agent_executor.invoke(input={
        "input": """Generate and save in a folder called qrcode under current working direction 5 QRCodes that
         point to https://www.linkedin.com/in/weifeng-hu-7b7364105/. If the folder does not exist, create it"""
    })


if __name__ == "__main__":
    main()