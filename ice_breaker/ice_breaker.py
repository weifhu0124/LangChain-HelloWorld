from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


information = """
Jeffrey Preston Bezos (/ˈbeɪzoʊs/ BAY-zohss;[2] né Jorgensen; born January 12, 1964) is an American businessman best known as the founder, executive chairman, and former president and CEO of Amazon, the world's largest e-commerce and cloud computing company. According to Forbes, as of May 2025, Bezos's estimated net worth exceeded $220 billion, making him the third richest person in the world.[3] He was the wealthiest person from 2017 to 2021, according to Forbes and the Bloomberg Billionaires Index.[4][5]
"""


if __name__ == "__main__":
    print("hello world")
    load_dotenv()

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": information})
    print(res)
