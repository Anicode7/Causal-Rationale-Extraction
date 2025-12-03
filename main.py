import os
# Must be set BEFORE importing transformers or gliner
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import llm_handler
import json
import embedder
import graph_operator
from datetime import datetime
import time
import graph_operator
import os 
import pandas as pd
import graph_gen

QUERIES_FILE = os.path.join(os.getcwd(),"data","queries_data.csv")



print("1 DATA LOADING-------------------------------------------------------------------")
now = datetime.now()
print(now.time())
    #transcript_text = json.load(f)["Hotel"][2]

#queries_data = pd.read_csv(QUERIES_FILE)

curr_query = "why are cancellation policies and fees causing customer frustration?"
curr_query_domain = "Hotel"
curr_query_intent = "Cancellation Policies"
print("CURRENT QUERY IS ", curr_query)

print("2 FINDING TOP 20 TRANSCRIPTS && CREATING GRAPH-------------------------------------------------------------------")


graph_gen = graph_gen.generate_json_graph(query=curr_query,domain=curr_query_domain,intent=curr_query_intent,data_path=os.path.join(os.getcwd(),"data","final_annotated_dataset.json"))


print("3 GETTING CAUSAL EDGES FROM SUBGRAPH AND ANSWER FROM LLM -------------------------------------------------------------------")

ans , _ = graph_operator.get_ans(curr_query)

print("-"*200)
print("FINAL ANSWER IS :\n ", ans)

