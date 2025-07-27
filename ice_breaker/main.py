from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from ice_breaker.agents.linkedin_lookup_agent import lookup as linkedin_lookup
from ice_breaker.third_parties.linkedin import scrape_linkedin_profile


def ice_break_with(name: str) -> str:
    linkedin_url = linkedin_lookup(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_url)

    summary_template = """
        given the LinkedIn information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    chain = summary_prompt_template | llm
    res = chain.invoke(input={"information": linkedin_data})
    return res


if __name__ == "__main__":
    load_dotenv()
    print(ice_break_with("Weifeng Hu AWS"))
