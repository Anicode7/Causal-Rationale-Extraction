# import os
# import json
# CURR_DIR = os.getcwd()
# JSON_FILE = os.path.join(CURR_DIR,"output","graph_with_metadata_embedded.json")

# with open(JSON_FILE, "r") as f:
#     graph = json.load(f)


# formatted_graph = {"nodes": {},"edges": []}
# for node in graph["nodes"]:
#     curr_id = node["id"]
#     formatted_graph["nodes"][curr_id] = node
# formatted_graph["edges"] = graph["edges"]
    
# with open(os.path.join(CURR_DIR,"output","formatted_graph_embedded_sarvesh.json"), "w") as f:
#     json.dump(formatted_graph, f, indent=4)