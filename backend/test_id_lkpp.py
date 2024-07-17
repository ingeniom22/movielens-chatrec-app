import json
import pandas as pd


with open("data/penyedia_id_mappings.json", "r") as json_file:
    penyedia_id_mappings = json.load(json_file)

print(penyedia_id_mappings)