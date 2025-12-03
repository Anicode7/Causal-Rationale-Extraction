import os
import json
import time
from neo4j import GraphDatabase
from networkx import edges
import llm_handler
from collections import deque, defaultdict
import copy
import statistics
import embedder
CURR_DIR = os.getcwd()
# --- CONFIGURATION ---
URI = "bolt://localhost:7687"
AUTH = ("", "") 



class MemgraphRAG:
    def __init__(self,JSON_FILE = os.path.join(CURR_DIR,"output","graph_with_metadata_embedded.json")):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
        self.llm = llm_handler.llm()
        self.JSON_FILE = JSON_FILE
        with open(self.JSON_FILE, 'r', encoding='utf-8') as f:
            self.init_graph = json.load(f)

    def close(self):
        self.driver.close()

    def setup_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            try: session.run("DROP VECTOR INDEX dialogue_vector_index")
            except: pass

            session.run("""
                CREATE VECTOR INDEX dialogue_vector_index ON :DialogueNode(embedding) 
                WITH CONFIG { "dimension": 768, "capacity": 10000, "metric": "cos" }
            """)
        print("✅ Database reset and index created.")

    def ingest_data(self, JSON_FILE=None):
        if JSON_FILE is None:
            JSON_FILE = self.JSON_FILE
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = self.init_graph
        if not os.path.exists(JSON_FILE):
            print(f"❌ File {JSON_FILE} not found.")
            return


        #print(data)
        self.graph = copy.deepcopy(data)
        for node in self.graph["nodes"]:
            node.pop("embedding", None)
        # print(self.graph)
        # print(self.graph["nodes"][0])
        print(f"📥 Ingesting {len(data.get('nodes', []))} nodes...")

        with self.driver.session() as session:
            # 1. INGEST NODES
            for node in data.get("nodes", []):
                id = f"{node.get('id')}"
                
                # Extract Lists
                turn_texts = node.get("utterance", [])
                turn_sources = node.get("utterance_sources", [])
                emb = node.get("embedding")
                
                
                # Flatten text for search
                flat_text = " ".join(turn_texts) if isinstance(turn_texts, list) else str(turn_texts)

                if id and emb:
                    session.run("""
                        MERGE (n:DialogueNode {id: $id})
                        SET n.text = $flat_text, 
                            n.utterance = $utterance, 
                            n.embedding = $emb,
                            n.escalation_level = $escalation_level,
                            n.churn_risk_score = $churn_risk_score,
                            n.empathy_score = $empathy_score,
                            n.escalation_risks = $escalation_risks,
                            n.intents_emotions = $intents_emotions,    
                            n.speaker = $speaker
                    """, id=id, flat_text=flat_text, utterance=turn_texts, 
                         emb=emb, escalation_level=node.get("escalation_level"),
                         churn_risk_score=node.get("churn_risk_score"),
                         empathy_score=node.get("empathy_score"),
                         escalation_risks=node.get("escalation_risks"),
                         intents_emotions=node.get("intents_emotions"),
                         speaker=node.get("speaker")
                    )

            # 2. INGEST EDGES (Loop logic removed)
            edges = data.get("edges", [])
            print(f"🔗 Connecting {len(edges)} edges...")
            
            for link in edges:
                src = f"{link.get("source")}"
                dst = f"{link.get("target")}"
                
                #print(src, "->", dst)
                session.run("""
                    MATCH (source:DialogueNode {id: $source_uid})
                    MATCH (target:DialogueNode {id: $target_uid})
                    MERGE (source)-[r:TRANSITION]->(target)
                """, source_uid=src, target_uid=dst)

    def get_landing_point(self, query_embedding, threshold=0.0, k=2):
        vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        query = f"""
            CALL vector_search.search('dialogue_vector_index', {k}, $vec)
            YIELD node, similarity
            WITH node, similarity
            WHERE similarity > $th
            RETURN node.id AS id, 
            node.utterance AS utterance, 
            node.escalation_level AS escalation_level,
            node.churn_risk_score AS churn_risk_score,
            node.empathy_score AS empathy_score,
            node.escalation_risks AS escalation_risks,
            node.intents_emotions AS intents_emotions,
            node.speaker AS speaker,
            similarity
        """
        
        with self.driver.session() as session:
            t = [r.data() for r in session.run(query, vec=vector, th=threshold)]
            #print(t)
            return t

    def gen_subgraph(self, nodes, max_depth=2):
        """
        Fetches the causal chain: (Cause) -> ... -> (Landing)
        """
        # We look BACKWARDS from the landing node to find causes
        query = """
        MATCH (target {id: $id})<-[*0..($max_depth)]-(n)
        RETURN DISTINCT {
            id: n.id,
            utterance: n.utterance,
            escalation_level: n.escalation_level,
            churn_risk_score: n.churn_risk_score,
            empathy_score: n.empathy_score,
            escalation_risks: n.escalation_risks,
            intents_emotions: n.intents_emotions,
            speaker: n.speaker
        } AS node_data;
        """

        query_edges = """
        MATCH (n)-[e]->(m)
        WHERE n.id IN $node_ids AND m.id IN $node_ids
        RETURN n.id AS source, 
        m.id AS target
        """

        output_data = []
        with self.driver.session() as session:
            for n in nodes:
                l_id = n['id']
                
                # Run query
                results = session.run(query, id=l_id,max_depth=max_depth)
                
                # FIX 3: Clean list comprehension. 
                # Don't append generators to the main list.
                # 'record["node_data"]' extracts the map projection we created in Cypher.
                subgraph_nodes = [record["node_data"] for record in results]
                
                print(f"✅ Found {len(subgraph_nodes)} nodes in chain for {l_id}.")
                ids = []
                for node in subgraph_nodes:
                    ids.append(node["id"])
                edges_result = session.run(query_edges, node_ids=ids)
                edges = [record.data() for record in edges_result]
                #print(edges)
                #print(uids)
                # Structure the final object
                output_data.append({
                    "landing_node_id": l_id,
                    "similarity_score": n['similarity'],
                    "landing_node_text": n['utterance'],
                    "causalgraph_nodes": subgraph_nodes,
                    "causalgraph_edges": edges
                })  

        # Save to file
        with open("subgraph.json", "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
            print(f"✅ Saved results to 'subgraph.json'")
            
        return output_data
    
    def get_causal_edge(self,edge):
        node_1_id = edge["source"]
        node_2_id = edge["target"]
        node_1 = self.fetch_metadata(node_1_id)
        node_2 = self.fetch_metadata(node_2_id)

        #print("NODE 1 IS ", node_1)
        causal_info = self.llm.generate_edge(node_1,node_2)

        #print(causal_info)
        return causal_info
    
    def search_for_edges_in_subgraph(self,subgraph,target_node):
        edges_found = []
        for edge in subgraph["causalgraph_edges"]:
            if edge["target"] == target_node:
                edges_found.append(edge)
        return edges_found
    
    def fetch_metadata(self,node_id):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:DialogueNode {id: $id})
                RETURN n.escalation_level AS escalation_level,
                       n.churn_risk_score AS churn_risk_score,
                       n.empathy_score AS empathy_score,
                       n.escalation_risks AS escalation_risks,
                       n.intents_emotions AS intents_emotions,
                       n.speaker AS speaker,
                       n.utterance AS utterance
            
            """, id=node_id)
            record = result.single()
            if record:
                return {
                    "escalation_level": record["escalation_level"],
                    "churn_risk_score": record["churn_risk_score"],
                    "empathy_score": record["empathy_score"],
                    "escalation_risks": record["escalation_risks"],
                    "intents_emotions": record["intents_emotions"],
                    "speaker": record["speaker"],
                    "utterance": record["utterance"],
                    "id": node_id
                }
            else:
                return {}


    def create_causal_chains(self, subgraphs, landing_nodes, threshold=0.1):
        ct = 0
        all_final_chains = [] # Store chains from ALL subgraphs here
        combined_subgraphs = []
        # Loop through each subgraph found by Cypher
        for i in range(len(subgraphs)):
            curr_subgraph = subgraphs[i]
            for edge in curr_subgraph["causalgraph_edges"]:
                ct += 1
        print("API CALLS MADE ARE ", ct)
        for i in range(len(subgraphs)):
            curr_subgraph = subgraphs[i]
            
            # --- STEP 1: ENRICHMENT & INDEXING ---
            adj_list = defaultdict(list)
            #print(f"   > Calculating probabilities for subgraph {i}...")

            for edge in curr_subgraph["causalgraph_edges"]:
                causal_info = self.get_causal_edge(edge) 
                prob = causal_info.get("probability", 0.0)
                ct+=1
                if prob > 0.1: 
                    rich_edge = {
                        "source": edge["source"],
                        "target": edge["target"],
                        "local_probability": prob, # Stored as 'local_probability'
                        "explanation": causal_info.get("explanation", "")
                    }
                    adj_list[edge["target"]].append(rich_edge)
                    combined_subgraphs.append(rich_edge)

            # --- STEP 2: SETUP NODES & SCORES ---
            node_lookup = {n["id"]: n for n in curr_subgraph["causalgraph_nodes"]}
            scores = {}
            
            # Initialize all scores to 0.0
            for node_data in curr_subgraph["causalgraph_nodes"]:
                scores[node_data["id"]] = 0.0

            # --- STEP 3: INITIALIZE QUEUE ---
            queue = []
            for l_id in landing_nodes:
                # l_id is a String. We check if it exists in this subgraph's lookup.
                if l_id in node_lookup:
                    scores[l_id] = 1.0
                    start_node_obj = node_lookup[l_id]
                    # Queue: [Current_ID, Score, Path_List]
                    queue.append([l_id, 1.0, [start_node_obj]])

            # --- STEP 4: TRAVERSAL (BFS) ---
            subgraph_chains = []
            #print(adj_list)
            while queue:
                current = queue.pop(0)
                current_id = current[0]
                current_score = current[1]
                current_path = current[2]

                # Lookup edges in our Enriched Adjacency List
                incoming_edges = adj_list[current_id]

                for edge in incoming_edges:
                    # Backward traversal: Target <- Source
                    next_node_id = edge["source"] 
                    
                    if next_node_id not in node_lookup:
                        continue
                    
                    next_node_obj = node_lookup[next_node_id]
                    
                    # FIX 1: Use the correct key 'local_probability'
                    local_prob = edge["local_probability"] 
                    
                    new_cumulative_prob = current_score * local_prob

                    if new_cumulative_prob < threshold:
                        continue
                    
                    if new_cumulative_prob > scores.get(next_node_id, 0.0):
                        scores[next_node_id] = new_cumulative_prob
                        
                        new_path = current_path + [edge, next_node_obj]
                        
                        queue.append([next_node_id, new_cumulative_prob, new_path])
                        
                        subgraph_chains.append({
                            "chain": new_path,
                            "cumulative_probability": new_cumulative_prob
                        })

            # Sort and pick top 5 for THIS subgraph
            subgraph_chains = sorted(subgraph_chains, key=lambda x: x["cumulative_probability"], reverse=True)[:5]
            
            # Add to the global list
            all_final_chains.extend(subgraph_chains)
        with open("enriched_subgraph.json", "w", encoding='utf-8') as f:
            json.dump({
                "edges": combined_subgraphs
            }, f, indent=4)
            print(f"✅ Saved enriched subgraph to 'enriched_subgraph.json'")
        # FIX 3: Return AFTER the loop finishes
        # Global Sort across all subgraphs
        all_final_chains = sorted(all_final_chains, key=lambda x: x["cumulative_probability"], reverse=True)[:5]
        #print(all_final_chains)


        
        return all_final_chains
    

    def stringify_chain(self,path_list):
        """
        Converts a mixed list of [Node, Edge, Node] into a readable string.
        """
        narrative_parts = []
        
        for item in path_list:
            if "explanation" in item:
                reason = item["explanation"]
                prob = item.get("local_probability", 0)
                # Add an arrow with the explanation
                narrative_parts.append(f"\n   ⬇️ CAUSES ({prob:.2f}): {reason} ⬇️\n")

            else:
                # It's a node. Extract useful info.
                node_id = item.get("id")
                text = item.get("utterance", "")
                # Handle list of utterances if necessary
                if isinstance(text, list): 
                    text = " ".join(text)
                    
                # You can add other metadata here like Sentiment or Risk
                escalation_level = item.get("escalation_level", "N/A")
                escalation_risk = item.get("escalation_risks", "N/A")
                churn_risk = item.get("churn_risk_score", "N/A")
                empathy_score = item.get("empathy_score", "N/A")
                dialogue_acts = item.get("dialogue_acts", "N/A")
                intents_emotions = item.get("intents_emotions", "N/A")
                speaker = item.get("speaker", "N/A")
                action_type = item.get("action_type", "N/A")
                
                narrative_parts.append(f"   [STATE {node_id}]: '{text}' (ESCALATION_LEVEL: {escalation_level}) , ESCALATION_RISK: {escalation_risk}, CHURN_RISK: {churn_risk}, EMPATHY_SCORE: {empathy_score}, DIALOGUE_ACTS: {dialogue_acts}, INTENTS_EMOTIONS: {intents_emotions}, SPEAKER: {speaker}, ACTION_TYPE: {action_type})")

        return "".join(narrative_parts)


    
def get_ans(query_text,UNEMBEDDED_JSON_FILE=os.path.join(CURR_DIR,"output","graph_with_metadata.json")):

    emb = embedder.embed_sentence(query_text)
    with open(UNEMBEDDED_JSON_FILE, 'r', encoding='utf-8') as f:
        unembedded_graph = json.load(f)
    embedder.embed_transcript(unembedded_graph)
    db = MemgraphRAG()
    EMBEDDED_JSON_FILE = os.path.join(CURR_DIR,"output","graph_with_metadata_embedded.json")
    db.setup_database()
    db.ingest_data(EMBEDDED_JSON_FILE)

    print("⏳ Waiting 2 seconds for vector index...")
    time.sleep(2)

    landing_results = db.get_landing_point(emb, threshold=0.0)
    #print(landing_results)
    print(f"🎯 Found {len(landing_results)} landing points.")

    with open ("landing_points.json", "w", encoding='utf-8') as f:
        json.dump(landing_results, f, indent=4)
        print(f"✅ Saved landing points to 'landing_points.json'")

    subgraphs = db.gen_subgraph(landing_results, max_depth=2)
    print(f"🎯 Found SUBGRAPHS")
    #
    # print(subgraphs)

    chains = db.create_causal_chains(subgraphs, [n["id"] for n in landing_results])


    print(f"🎯 Created CAUSAL CHAINS")
    for chain in chains:
        chain.pop("local_probability", None)

    with open ("causal_chains.json", "w", encoding='utf-8') as f:
        json.dump(chains, f, indent=4)
        print(f"✅ Saved causal chains to 'causal_chains.json'")
    formatted_chains_text = []
    for i, chain_wrapper in enumerate(chains):
        # chain_wrapper is the dict with {"chain": [...], "cumulative_probability": ...}
        path_str = db.stringify_chain(chain_wrapper["chain"])
        score = chain_wrapper["cumulative_probability"]
        
        formatted_chains_text.append(f"### PATHWAY {i+1} (Confidence: {score:.2f})\n{path_str}")
    full_context_string = "\n\n".join(formatted_chains_text)
    ans = db.llm.answer_query_causal(query_text,full_context_string)
    
    return ans , db