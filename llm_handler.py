import os
import json
import ollama
from pydantic import BaseModel

class EdgeSchema(BaseModel):
    causal: bool
    probability: float
    explanation: str

class llm:
    def __init__(self, model="llama3.1"): 
        self.model = model
        
        # --- OPTIMIZED BATCH PROMPT ---
        self.batch_prompt = """
You are a causal reasoning engine. Analyze the BATCH of node pairs.
Determine if 'SOURCE' caused 'TARGET'.

SCORING:
0.0: No link/Reaction unrelated.
0.1-0.5: Weak link/Topic shift.
0.6-1.0: Strong causal link (Direct answer/complaint).

OUTPUT:
Return ONLY a JSON object. No markdown.
{
    "results": [
        { "id": <INT>, "causal": <BOOL>, "probability": <FLOAT>, "explanation": "<SHORT_STRING>" }
    ]
}
"""


    def query_splitter(self,query):
        prompt = f"""You are an expert NLU (Natural Language Understanding) router. Your goal is to map a user's input query to the most relevant (Domain, Intent) pairs from a provided list and optimize the query for search.

### Instructions:
1. **Analyze:** Read the user's input query.
2. **Select:** Compare the input against the "Allowed Domain-Intent List" provided below. Select the best matches (maximum 3, sorted by relevance). If the query is unambiguous, select only 1. If no match is found, return an empty 'matches' array: [].
3. **Reform:** For each selected pair, generate a 'reformed_query'. This query must strip conversational filler and inject specific technical keywords or terminology relevant to that specific (Domain, Intent) to assist downstream search retrieval.
4. **Output:** Return the result in the specified JSON format only. DO NOT include any introductory or explanatory text outside of the JSON block.

### Allowed Domain-Intent List (37 Pairs):
[
  ("Banking", "Credit Limit Requests"),
  ("Banking", "Fee Complaints"),
  ("Banking", "Fraud Alerts"),
  ("Banking", "Loan Application"),
  ("Banking", "Product Comparison"),
  ("Banking", "Refund Delays"),
  ("Flight", "Cross Brand Mentions"),
  ("Flight", "Delay Management"),
  ("Flight", "Loyalty Program"),
  ("Flight", "Price Sensitivity"),
  ("Flight", "Refund Policy"),
  ("Flight", "Urgency & Stress"),
  ("Hotel", "Booking Errors"),
  ("Hotel", "Brand Loyalty"),
  ("Hotel", "Cancellation Policies"),
  ("Hotel", "Discounts & Promotions"),
  ("Hotel", "Service Complaints"),
  ("Hotel", "Upgrade Requests"),
  ("Insurance", "Claims & Refunds"),
  ("Insurance", "Competitor Comparison"),
  ("Insurance", "Customer Trust"),
  ("Insurance", "Feature Understanding"),
  ("Insurance", "Policy Renewal"),
  ("Insurance", "Sales Effectiveness"),
  ("Insurance", "Upselling Strategy"),
  ("Retail", "Delivery Delays"),
  ("Retail", "Loyalty Program"),
  ("Retail", "Product Feedback"),
  ("Retail", "Product Returns"),
  ("Retail", "Replacement Vs Refund"),
  ("Telecom", "Churn Prediction"),
  ("Telecom", "Connectivity Complaints"),
  ("Telecom", "Feature Requests"),
  ("Telecom", "Network Outages"),
  ("Telecom", "Plan Upgrades"),
  ("Telecom", "Technical Support")
]

### Output Format (JSON):
{
  "matches": [
    {
      "domain": "String",
      "intent": "String",
      "reformed_query": "String",
      "reasoning": "Brief explanation of why this pair was chosen"
    }
  ]
}
"""
        
        user_content_str = f"""
This is the User Query:
{query}
Now generate the output as per the instructions above.
"""
        
        response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": user_content_str}
                ],
                format="json",
                options={
                    "num_gpu": 99,     
                    "num_ctx": 16384,   
                    "temperature": 0.0,
                    "num_predict": -1,   # Unlimited output tokens
                    "num_batch": 2048    # <--- THIS IS THE CRITICAL FIX FOR SPEED
                },
                keep_alive="24h" 
            )

        content = response['message']['content']
        parsed_response = json.loads(content)
        return parsed_response.get("matches", [])


    def generate_batch_edges(self, edge_batch):
        # 1. Construct the batch string
        user_content_str = "ANALYZE THIS BATCH:\n"
        
        for item in edge_batch:
            # Minimal data representation to save tokens
            source = item['source']
            target = item['target']
            
            # Helper to safely truncate text
            def get_s(n, k): return str(n.get(k, ''))[:150] 

            s_text = f"Spk:{source.get('speaker')} Utt:{get_s(source, 'utterance')} Emo:{get_s(source, 'intents_emotions')}"
            t_text = f"Spk:{target.get('speaker')} Utt:{get_s(target, 'utterance')} Emo:{get_s(target, 'intents_emotions')}"

            user_content_str += f"ID:{item['id']} | SRC:[{s_text}] -> TGT:[{t_text}]\n"

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.batch_prompt},
                    {"role": "user", "content": user_content_str}
                ],
                format="json",
                options={
                    "num_gpu": 99,     
                    "num_ctx": 16384,   
                    "temperature": 0.0,
                    "num_predict": -1,   # Unlimited output tokens
                    "num_batch": 2048    # <--- THIS IS THE CRITICAL FIX FOR SPEED
                },
                keep_alive="24h" 
            )

            content = response['message']['content']
            parsed_response = json.loads(content)
            return parsed_response.get("results", [])

        except Exception as e:
            print(f"⚠️ Batch inference failed: {e}")
            return []

    # ... (Rest of the class methods: answer_query, answer_query_causal remain unchanged)
    def answer_query(self, query, context):
        sys_prompt = f'''
        Your task is to perform causal analysis given the causal chains related to the query and a query.
        In each causal chain you will find a series of causes and effects with explanations and probabilities.
        Your goal is to analyze these chains to provide a concise answer to the user's query.
        DO NOT MENTION the cumulative probabilities in your answer. However you can mention the implications of properties and metdata such as empathy score, escalation risk etc of high or low probability chains in your reasoning.
        Give a short concise explanation using ONLY the context to answer the query.
        '''

        curr_prompt = f'''Query -- {query}
        Causal Chains -- {context}
        Give your answer now :'''
        
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": curr_prompt}
            ],
            options={
                "num_gpu": 99,
                "num_ctx": 4096
            },
            keep_alive="5m"
        )

        ans = response['message']['content']
        return ans

    def answer_query_causal(self, query, chains):
        sys_prompt = f'''
        You are an expert Root Cause Analysis AI.
        Your task is to perform causal analysis given the causal chains related to the query and a query.
        You will be given a causal chain in the form of NODE , EDGE , NODE ... format. For each node, you will have metadata such as utterance , empathy score, escalation risk, churn risk , intent_emotions etc. For each edge, you will have an explanation and a cumulative probability score.
        The edges represent the causal relationships between the nodes.
        Your goal is to analyze these chains to provide a concise answer to the user's query.

        INSTRUCTIONS:
        1. REVIEW: Read the provided chains (ordered: Cause -> Effect).
        2. ANALYZE METADATA : Analyze all the metadata such as utterance , empathy score, escalation risk, churn risk , intent_emotions etc in the nodes of the chains to get better context on the causal relationships.
        3. WEIGH: Pay strict attention to the 'cumulative_probability' score. Trust high-probability chains (0.7+) over low ones.
        4. ANSWER: Provide a concise, direct answer to the query based ONLY on this evidence. Do not invent facts.

        OUTPUT_FORMAT -- DO NOT MENTION the cumulative probabilities in your answer. However you can mention the implications of properties and metdata such as empathy score, escalation risk etc of high or low probability chains in your reasoning.
        Give a short concise explanation using ONLY the causal chains to answer the query
        '''

        curr_prompt = f'''Query -- {query}
        CAUSAL CHAINS--- 
        {chains}
        
        Give your answer now :'''

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": curr_prompt}
            ],
            options={
                "num_gpu": 99,
                "num_ctx": 8192 
            },
            keep_alive="5m"
        )

        ans = response['message']['content']
        return ans