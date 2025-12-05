from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import statistics
# run embedder first then graph_formatter
CURR_DIR = os.getcwd()
JSON_FILE = os.path.join(CURR_DIR,"output","graph_with_metadata.json")
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert NumPy integers to standard ints
        if isinstance(obj, np.integer):
            return int(obj)
        # Convert NumPy floats to standard floats
        if isinstance(obj, np.floating):
            return float(obj)
        # Convert NumPy arrays to lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

model = SentenceTransformer("all-mpnet-base-v2")
def embed_node(node):   
    # embeddings = model.encode(node["utterances"])
    # avg_embedding =np.mean(np.array(embeddings), axis=0)
    text = ""
    text += node["dialogue_acts"][0] + " "
    text += node["action_type"][0] + " "
    text += node["intents_emotions"][0] + " "
    text += node["utterance"][0]
    emb = model.encode(text.strip())
    return emb

def embed_transcript(graph):
    for node in graph["nodes"]:
        emb = embed_node(node)
        node.pop("escalation_level_std", None)
        node.pop("escalation_level_min", None)
        node.pop("escalation_level_max", None)
        node.pop("churn_risk_score_std", None)
        node.pop("churn_risk_score_min", None)
        node.pop("churn_risk_score_max", None)
        node.pop("empathy_score_std", None)
        node.pop("empathy_score_min", None)
        node.pop("empathy_score_max", None)
        node.pop("escalation_reasons_tags", None)
        node.pop("utterance_sources", None)
        node.pop("utterances",None)
        node.pop("num_turns_in_cluster",None)
        node.pop("cluster_label",None)
        node.pop("escalation_reason_tags",None)
        node["escalation_risks"] = statistics.mean(node.get("escalation_risks"))
        node["dialogue_acts"] = node.get("dialogue_acts")[0]
        node["action_type"] = node.get("action_type")[0]
        node["intents_emotions"] = node.get("intents_emotions")[0]
        
        node["embedding"] = emb
        with open(os.path.join(CURR_DIR,"output","graph_with_metadata_embedded.json"), "w") as outfile:
            json.dump(graph, outfile, indent=4,cls=NpEncoder)  

def embed_sentence(sentence):
    return model.encode(sentence)

if __name__ == "__main__":
    with open(os.path.join(CURR_DIR,"output","graph_with_metadata.json"), "r") as f:
        graph = json.load(f)
    embed_transcript(graph)