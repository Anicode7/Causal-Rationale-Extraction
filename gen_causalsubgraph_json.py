import json 
import os


with open(os.path.join(os.getcwd(), "subgraph.json"), 'r') as file:
    data = json.load(file)

temp = {}
temp["nodes"] = []
temp["edges"] = []

for node in data:
    temp["nodes"].append(node["causalgraph_nodes"])

with open(os.path.join(os.getcwd(), "edges_computed.json"), 'r') as file:
    edge_data = json.load(file)

temp["edges"] = edge_data

with open(os.path.join(os.getcwd(),"causal_subgraph.json"), 'w') as file:
    json.dump(temp, file, indent=4, ensure_ascii=False)
# with open(os.path.join(os.getcwd(), "edges_computed.json"), 'r') as file:
#     edge_data = json.load(file)
# print(len(edge_data))