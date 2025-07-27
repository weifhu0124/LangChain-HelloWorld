import os
import requests
from dotenv import load_dotenv

from ice_breaker.constants import LINKEDIN_PROFILE

load_dotenv()


def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = True):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""

    # https://app.scrapin.io/lookup/persons
    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/weifhu0124/a35f81fa5948ff9dfd0b71ef4bb63c1d/raw/948c445c6a14f65c0999eaa2c260f45d542e0e38/weifenhu-linkedIn.json"
        response = requests.get(
            linkedin_profile_url,
            timeout=10,
        )
    else:
        # scrapin IO
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.environ["SCRAPIN_API_KEY"],
            "linkedInUrl": linkedin_profile_url,
        }
        response = requests.get(
            api_endpoint,
            params=params,
            timeout=10,
        )

    data = response.json().get("person")
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None) and k not in ["certifications"]
    }

    return data


if __name__ == "__main__":
    print(
        scrape_linkedin_profile(
            linkedin_profile_url=LINKEDIN_PROFILE
        ),
    )
