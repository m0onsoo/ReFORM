import json ,os
from openai import OpenAI
from pydantic import BaseModel
import random
import numpy as np

# Get GPT API key
OpenAI.api_key = "" # YOUR OPENAI API_KEY
client = OpenAI()

# output form
class UserProfile(BaseModel):
    cuisine_type: str=None
    flavor: str=None
    atmosphere: str=None
    price_tolerance: str=None
    companion: str=None
    time: str=None
    wait_time: str=None

# user
factor_descriptions = {
    "cuisine type": "The specific type of food offered appeals to user's preferences and dietary restrictions.",
    "flavor": "The specific taste characteristics that user prefers.",
    "atmosphere":"User's preferences about the atmosphere of restaurant or staff.",
    "price tolerance": "Affordability and value for money when user chooses a restaurant.",
    "companion": "Who does user usually go to restaurants with",
    "time": "What time does user prefer to go a restaurant(e.g., breakfast, lunch, dinner, brunch, morning tea,afternoon tea, etc.)? ",
    "wait time": "Identify user preferences for waiting durations at restaurants.",
}
system_prompt = f"You are a helpful restaurant review analyst.\nI'll give you reviews that users have written to the restaurant.\nI would be glad if you extract 3 to 5 values about what this user prefer based on the reviews following the factors described below.\n {json.dumps(factor_descriptions,indent=4)}).\n In \'flavor\' factor, Exclude subjective terms(e.g., delicious) and the name of dishes(e.g., pasta,fried chicken, fish and chips).\nIf you can't find specific things from these factors, return 'None' to the output key.\nGive me the output as a noun"

def gen_respone(interactions):
    if len(interactions) > 100:
        interactions = random.sample(interactions, 100)
    
    response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{interactions}"},
            ],
            max_tokens=5000,
            temperature=0.1,
            response_format=UserProfile,
        )

    response = json.loads(response.choices[0].message.content)
    return interactions, response




# ---------------------------------------------------------
input_path = './generation/profile/user/user_reviews.json'

# Load input file
with open(input_path, "r") as f:
    input_data = json.load(f)

indexs = len(input_data)
picked_id = np.random.choice(indexs, size=1)[0]

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

print(Colors.GREEN + "Generating Profile for User" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Input Reviews are:\n" + Colors.END)
interactions, response = gen_respone(input_data[picked_id]['interactions'])
print(json.dumps(interactions[:5], indent=4))
print("---------------------------------------------------\n")
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(json.dumps(response, indent=4))