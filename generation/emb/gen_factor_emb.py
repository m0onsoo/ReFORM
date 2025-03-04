from transformers import AutoTokenizer, BertModel
from tqdm import tqdm
import json, os
import numpy as np
import argparse

'''
python generation/emb/gen_factor_emb.py --name usr
python generation/emb/gen_factor_emb.py --name itm
'''

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="Need to specify 'usr' or 'itm' that you want to encode")
args = parser.parse_args()

# BERT
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")

def get_bert_emb(factors):
    embs = []
    for factor_value in factors.values():
        factor_cleaned = factor_value.replace(',', ' ')
        
        inputs = tokenizer(factor_cleaned, return_tensors="pt")
        outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        valid_embeddings = last_hidden_states[:, 1:-1, :]

        emb = valid_embeddings.mean(dim=1).detach().numpy().squeeze()
        embs.append(emb)

    return np.array(embs)


input_file = './generation/emb/{}_profiles.json'.format(args.name)

if not os.path.exists(input_file):
    raise Exception("Please check file name or directory again.")
with open(input_file, "r") as f:
    profiles = json.load(f)


class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'
    YELLOW = '\033[93m'

print(Colors.GREEN + "Encoding Semantic Representation" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Profile is:\n" + Colors.END)
print(json.dumps(profiles[0]['model_response'], indent = 4))
print("---------------------------------------------------\n")
embs = get_bert_emb(profiles[0]['model_response'])
print(Colors.GREEN + "Encoded Semantic Representation Shape:" + Colors.END)
print(embs.shape)