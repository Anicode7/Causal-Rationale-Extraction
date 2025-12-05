import os
import json
import time
from neo4j import GraphDatabase
import llm_handler
from collections import deque, defaultdict
import copy
import embedder
from concurrent.futures import ThreadPoolExecutor, as_completed

CURR_DIR = os.getcwd()
# --- CONFIGURATION ---
URI = "bolt://localhost:7687"
AUTH = ("", "") 

class MemgraphRAG:
    def __init__(self, JSON_FILE=os.path.join(CURR_DIR, "output", "graph_with_metadata_embedded.json")):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
        self.llm = llm_handler.llm()
        self.JSON_FILE = JSON_FILE
        # Only load if file exists to prevent errors on fresh start
        if os.path.exists(self.JSON_FILE):
            with open(self.JSON_FILE, 'r', encoding='utf-8') as f:
                self.init_graph = json.load(f)
        else:
            self.init_graph = {}

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
        print("Database reset and index created.")

    def ingest_data(self, JSON_FILE=None):
        if JSON_FILE is None:
            JSON_FILE = self.JSON_FILE
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = self.init_graph
        
        if not os.path.exists(JSON_FILE) and not data:
            print(f"File {JSON_FILE} not found or empty.")
            return

        self.graph = copy.deepcopy(data)
        print(f"Ingesting {len(data.get('nodes', []))} nodes...")

        with self.driver.session() as session:
            # 1. INGEST NODES
            for node in data.get("nodes", []):
                id = f"{node.get('id')}"
                
                turn_texts = node.get("utterance", [])
                emb = node.get("embedding")
                
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

            # 2. INGEST EDGES
            edges = data.get("edges", [])
            print(f"Connecting {len(edges)} edges...")
            
            for link in edges:
                src = f"{link.get('source')}"
                dst = f"{link.get('target')}"
                
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
            return t

    def gen_subgraph(self, nodes, max_depth=2):
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
                results = session.run(query, id=l_id, max_depth=max_depth)
                subgraph_nodes = [record["node_data"] for record in results]
                
                print(f"Found {len(subgraph_nodes)} nodes in chain for {l_id}.")
                ids = [node["id"] for node in subgraph_nodes]

                edges_result = session.run(query_edges, node_ids=ids)
                edges = [record.data() for record in edges_result]
                
                output_data.append({
                    "landing_node_id": l_id,
                    "similarity_score": n['similarity'],
                    "landing_node_text": n['utterance'],
                    "causalgraph_nodes": subgraph_nodes,
                    "causalgraph_edges": edges
                })  

        with open("subgraph.json", "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
            print(f"Saved results to 'subgraph.json'")
            
        return output_data
    
    def fetch_metadata(self, node_id):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:DialogueNode {id: $id})
                RETURN n.escalation_level AS escalation_level,
                       n.churn_risk_score AS churn_risk_score,
                       n.empathy_score AS empathy_score,
                       n.escalation_risks AS escalation_risks,
                       n.intents_emotions AS intents_emotions,
                       n.speaker AS speaker,
                       n.utterance AS utterance,
                       n.id AS id
            """, id=node_id)
            record = result.single()
            if record:
                return dict(record)
            else:
                return {}

    def create_causal_chains(self, subgraphs, landing_nodes, threshold=0.1):
        """
        UPDATED: Parallel Batch Processing.
        Uses ThreadPoolExecutor to saturate OLLAMA_NUM_PARALLEL slots.
        """
        all_final_chains = []
        combined_subgraphs = []
        
        # --- PHASE 1: COLLECT EDGES FOR BATCHING ---
        edges_to_process = []
        seen_pairs = set()

        print("Preparing edges for batch processing...")
        
        for i, curr_subgraph in enumerate(subgraphs):
            node_lookup = {n["id"]: n for n in curr_subgraph["causalgraph_nodes"]}
            
            for edge in curr_subgraph["causalgraph_edges"]:
                source_id = edge["source"]
                target_id = edge["target"]
                
                if source_id in node_lookup and target_id in node_lookup:
                    pair_key = f"{source_id} {target_id}"
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    edges_to_process.append({
                        "edge_ref": edge, # Reference to update IN-PLACE
                        "source": node_lookup[source_id],
                        "target": node_lookup[target_id],
                        "subgraph_index": i
                    })


        # --- PHASE 2: PARALLEL BATCH LLM PROCESSING ---
        BATCH_SIZE = 30 # Large batch for efficiency
        total_edges = len(edges_to_process)
        MAX_WORKERS = 4 # Match this with OLLAMA_NUM_PARALLEL
        
        print(f"Processing {total_edges} edges in batches of {BATCH_SIZE} using {MAX_WORKERS} Parallel Workers...")
        
        batches = [edges_to_process[i:i + BATCH_SIZE] for i in range(0, total_edges, BATCH_SIZE)]
        
        # Helper function for threading
        def process_batch_task(batch_data):
            llm_input = []
            for idx, item in enumerate(batch_data):
                llm_input.append({
                    "id": idx, # Relative ID
                    "source": item["source"],
                    "target": item["target"]
                })
            # Return both results and the original batch to map them back
            return self.llm.generate_batch_edges(llm_input), batch_data

        calls_made = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {executor.submit(process_batch_task, batch): batch for batch in batches}
            
            for future in as_completed(future_to_batch):
                calls_made += 1
                try:
                    results, original_batch = future.result()
                    
                    # Map results back to edge references
                    for res in results:
                        res_id = res.get("id")
                        if res_id is not None and isinstance(res_id, int) and 0 <= res_id < len(original_batch):
                            original_edge_dict = original_batch[res_id]["edge_ref"]
                            original_edge_dict["local_probability"] = res.get("probability", 0.0)
                            original_edge_dict["explanation"] = res.get("explanation", "")
                except Exception as exc:
                    print(f"Batch generation generated an exception: {exc}")

        print(f"LLM Processing complete. {calls_made} Batch calls made.")

        # --- PHASE 3: TRAVERSAL (BFS) ---
        for i, curr_subgraph in enumerate(subgraphs):
            adj_list = defaultdict(list)
            
            for edge in curr_subgraph["causalgraph_edges"]:
                prob = edge.get("local_probability", 0.0)
                
                if prob > 0.1:
                    rich_edge = {
                        "source": edge["source"],
                        "target": edge["target"],
                        "local_probability": prob,
                        "explanation": edge.get("explanation", "")
                    }
                    adj_list[edge["target"]].append(rich_edge)
                    combined_subgraphs.append(rich_edge)

            node_lookup = {n["id"]: n for n in curr_subgraph["causalgraph_nodes"]}
            scores = {n["id"]: 0.0 for n in curr_subgraph["causalgraph_nodes"]}

            queue = []
            for l_id in landing_nodes:
                if l_id in node_lookup:
                    scores[l_id] = 1.0
                    start_node_obj = node_lookup[l_id]
                    queue.append([l_id, 1.0, [start_node_obj]])

            subgraph_chains = []
            while queue:
                current = queue.pop(0)
                current_id = current[0]
                current_score = current[1]
                current_path = current[2]

                incoming_edges = adj_list[current_id]

                for edge in incoming_edges:
                    next_node_id = edge["source"]
                    
                    if next_node_id not in node_lookup:
                        continue
                    
                    next_node_obj = node_lookup[next_node_id]
                    local_prob = edge["local_probability"]
                    new_cumulative_prob = current_score * local_prob

                    if new_cumulative_prob < threshold:
                        continue
                    
                    if new_cumulative_prob > scores.get(next_node_id, 0.0):
                        scores[next_node_id] = new_cumulative_prob
                        new_path = current_path + [edge, next_node_obj]
                        queue.append([next_node_id, new_cumulative_prob, new_path])
                        
                        final_chronological_chain = new_path[::-1]
                        
                        subgraph_chains.append({
                            "chain": final_chronological_chain, 
                            "cumulative_probability": new_cumulative_prob
                        })
            subgraph_chains = sorted(subgraph_chains, key=lambda x: x["cumulative_probability"], reverse=True)[:5]
            all_final_chains.extend(subgraph_chains)


        try:
            with open("edges_computed.json", 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = [] 
        
        new_data = []
        for elem in combined_subgraphs:
            new_data.append({
                "source": self.fetch_metadata(elem["source"]),
                "target": self.fetch_metadata(elem["target"]),
                "local_probability": elem["local_probability"],
                "explanation": elem["explanation"]
            })
            
        if isinstance(existing_data, list):
            existing_data.extend(new_data)
        else:
            existing_data.update(new_data)

        with open("edges_computed.json", 'w') as file:
            json.dump(existing_data, file, indent=4)
            print(f"Saved edges to 'edges_computed.json'")

        all_final_chains = sorted(all_final_chains, key=lambda x: x["cumulative_probability"], reverse=True)[:5]
        
        return all_final_chains

    def stringify_chain(self, path_list):
        narrative_parts = []
        for item in path_list:
            if "explanation" in item:
                reason = item["explanation"]
                prob = item.get("local_probability", 0)
                narrative_parts.append(f"\n   ⬇️ CAUSES ({prob:.2f}): {reason} ⬇️\n")
            else:
                node_id = item.get("id")
                text = item.get("utterance", "")
                if isinstance(text, list): 
                    text = " ".join(text)
                escalation_level = item.get("escalation_level", "N/A")
                churn_risk = item.get("churn_risk_score", "N/A")
                empathy_score = item.get("empathy_score", "N/A")
                speaker = item.get("speaker", "N/A")
                narrative_parts.append(f"   [STATE {node_id}]: '{text}' (Speaker: {speaker}, Esc_Level: {escalation_level}, Churn: {churn_risk}, Empathy: {empathy_score})")
        return "".join(narrative_parts)


# REPLACE the existing get_ans function with this updated version:

def get_ans(query_text, queries_list, follow_up=0, UNEMBEDDED_JSON_FILE=os.path.join(CURR_DIR, "output", "graph_with_metadata.json")):

    # --- 1. HISTORY MANAGEMENT ---
    with open (os.path.join(os.getcwd(),"cached_convos" , "last_conversation_number.json"), 'r', encoding='utf-8') as f:
        conv_num_data = json.load(f)


    if follow_up == 0:
        # Reset history for new query
        conversation_history = []
        print("type is", type(conv_num_data["num"]))
        conv_num_data["num"] = int(conv_num_data["num"]) + 1

        with open (os.path.join(os.getcwd(),"cached_convos" , "last_conversation_number.json"), 'w', encoding='utf-8') as f:
            json.dump(conv_num_data, f, indent=4)
       # with open (os.path.join(os.getcwd(),"cached_convos" , "last_conversation_number.json"), 'w', encoding='utf-8') as f:


    HISTORY_FILE = os.path.join(os.getcwd(),"cached_convos",f"conversation_history_{conv_num_data["num"]}.json")
    conversation_history = []
    TRANSCRIPT_HISTORY_FILE = os.path.join(os.getcwd(),"cached_convos",f"transcript_history_{conv_num_data["num"]}.json")
    curr_transcripts_file = os.path.join(CURR_DIR, "data", "transcript_id_list.txt")
    with open(curr_transcripts_file, 'r', encoding='utf-8') as f:
        transcript_list = [line.strip() for line in f if line.strip()]
    with open(TRANSCRIPT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(transcript_list, f, indent=4)
        
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as hf:
                conversation_history = json.load(hf)
        except json.JSONDecodeError:
            conversation_history = []
            
    # Append current user query
    conversation_history.append({"role": "user", "content": query_text})

    # --- 2. EXISTING GRAPH LOGIC (UNCHANGED) ---

    emb = embedder.embed_sentence(query_text)
    emblist = []
    for i in queries_list:
        emblist.append(embedder.embed_sentence(i["reformed_query"]))
    
    if os.path.exists(UNEMBEDDED_JSON_FILE):
        with open(UNEMBEDDED_JSON_FILE, 'r', encoding='utf-8') as f:
            unembedded_graph = json.load(f)
        embedder.embed_transcript(unembedded_graph) 

    db = MemgraphRAG()
    EMBEDDED_JSON_FILE = os.path.join(CURR_DIR, "output", "graph_with_metadata_embedded.json")
    
    db.setup_database()
    db.ingest_data(EMBEDDED_JSON_FILE)

    print("Waiting 2 seconds for vector index...")
    time.sleep(2)

    landing_results = []
    for i in emblist:
        results = db.get_landing_point(i, threshold=0.0)
        landing_results.extend(results)
    print(f"Found {len(landing_results)} landing points.")

    with open ("landing_points.json", "w", encoding='utf-8') as f:
        json.dump(landing_results, f, indent=4)

    subgraphs = db.gen_subgraph(landing_results, max_depth=2)
    print(f"Found SUBGRAPHS")

    chains = db.create_causal_chains(subgraphs, [n["id"] for n in landing_results])
    print(f"Created CAUSAL CHAINS")

    for chain_wrapper in chains:
        for item in chain_wrapper["chain"]:
            if "explanation" in item:
                 item.pop("local_probability", None)

    with open ("causal_chains.json", "w", encoding='utf-8') as f:
        json.dump(chains, f, indent=4)
    
    formatted_chains_text = []
    for i, chain_wrapper in enumerate(chains):
        path_str = db.stringify_chain(chain_wrapper["chain"])
        score = chain_wrapper["cumulative_probability"]
        formatted_chains_text.append(f"### PATHWAY {i+1} (Confidence: {score:.2f})\n{path_str}")
    
    full_context_string = "\n\n".join(formatted_chains_text)
    
    # --- 3. UPDATED LLM CALL & HISTORY SAVE ---
    if follow_up > 0:
        print("Generating Answer (Follow-up Context)...")
        ans = db.llm.answer_followup_causal(query_text, full_context_string, conversation_history)
    else:
        print("Generating Answer (Fresh Context)...")
        ans = db.llm.answer_query_causal(query_text, full_context_string)
    
    # Save Assistant Response to History
    conversation_history.append({"role": "assistant", "content": ans})
    with open(HISTORY_FILE, 'w', encoding='utf-8') as hf:
        json.dump(conversation_history, hf, indent=4)

     
    
    return ans, db