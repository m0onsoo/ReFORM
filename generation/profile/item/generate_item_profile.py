import json ,os
from openai import OpenAI
from pydantic import BaseModel
import random
import numpy as np

# Get GPT API key
OpenAI.api_key = "" # YOUR OPENAI API_KEY
client = OpenAI()

# output form
class ItemProfile(BaseModel):
    cuisine_type: str=None
    flavor: str=None
    atmosphere: str=None
    price_tolerance: str=None
    companion: str=None
    time: str=None
    wait_time: str=None

# item
factor_descriptions = [
    "cuisine type - Identify the cuisine types served by the restaurant based on customer reviews. Focus on general categories (e.g., Italian, Mexican, Vegan). If multiple cuisines are mentioned, include all relevant categories.",
    "flavor - The specific tastes or aromas frequently mentioned by users when describing this restaurant's food.",
    "atmosphere - Describe the unique ambiance and characteristics of the restaurant, including specific details about decor, lighting, and the friendliness of the staff as mentioned in reviews.",
    "price tolerance - Users' comfort level with the restaurant's prices, including whether they find the cost reasonable for the experience.",
    "companion - Identify specific types of groups or people users mention dining with, such as a romantic date spot for couples, family-friendly gathering place, or a budget-friendly option for friends.",
    "time - Identify the preferred dining times users mention for this restaurant, such as 'breakfast,' 'lunch,' 'dinner,' or 'brunch.'",
    "wait time - Determine the wait time for getting a table at this restaurant. Convert any numerical durations (e.g., '20 minutes', '1 hour') into descriptive and qualitative text.",
]

# item
system_prompt = f"You will be provided with multiple reviews for a restaurant, and your task is to extract what users consider important when reviewing this restaurant based on the following factors ({', '.join(factor_descriptions)}).\nIf you can't find any of the specified factors in the reviews, return 'None' as the output.\nProvide 3 to 5 values and avoid general answers."

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
            response_format=ItemProfile,
        )

    response = json.loads(response.choices[0].message.content)
    return interactions, response




# ---------------------------------------------------------
input_path = './generation/profile/item/item_reviews.json'

# Load input file
with open(input_path, "r") as f:
    input_data = json.load(f)

indexs = len(input_data)
picked_id = np.random.choice(indexs, size=1)[0]

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

print(Colors.GREEN + "Generating Profile for item" + Colors.END)
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