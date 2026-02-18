import os
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file, mainly the GEMINI_API_KEY
load_dotenv()

#os.env gets the gemini api key that we just loaded into our enviornment
gemini_api_key = os.getenv("GEMINI_API_KEY")
#creates connection to our gemini api
client = genai.Client(api_key=gemini_api_key)

#function that takes in a string and returns the response from the gemini api, using the gemini-2.5-flash model
def response(content):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=content
    )
    return(response.text)