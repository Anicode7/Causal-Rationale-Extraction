import os
import json
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = ("", "") 
driver = GraphDatabase.driver(URI, auth=AUTH)



with open(os.path.join(os.getcwd(), "causal_subgraph.json"), 'r') as file:
    data = json.load(file)

with driver.session() as session:
    # 1. INGEST NODES
    for node in data.get("nodes", []):
        id = f"{node.get('id')}"
        
        turn_texts = node.get("utterance", [])

        
        flat_text = " ".join(turn_texts) if isinstance(turn_texts, list) else str(turn_texts)

        if id:
            session.run("""
                MERGE (n: Node {id: $id})
                SET n.text = $flat_text, 
                    n.utterance = $utterance, 
                    n.escalation_level = $escalation_level,
                    n.churn_risk_score = $churn_risk_score,
                    n.empathy_score = $empathy_score,
                    n.escalation_risks = $escalation_risks,
                    n.intents_emotions = $intents_emotions,    
                    n.speaker = $speaker
            """, id=id, flat_text=flat_text, utterance=turn_texts, 
                    escalation_level=node.get("escalation_level"),
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
        src = f"{link.get('source').get("id")}"
        dst = f"{link.get('target').get("id")}"
        prob = link.get("local_probability", 0.0)
        expl = link.get("explanation", "")
        session.run("""
            MATCH (source: Node{id: $source_uid})
            MATCH (target: Node{id: $target_uid})
            MERGE (source)-[r:TRANSITION]->(target)
                SET r.local_probability = $prob,
            r.explanation = $expl
        """, source_uid=src, target_uid=dst,prob =prob, expl=expl)