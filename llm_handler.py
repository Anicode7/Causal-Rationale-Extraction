import os
import json
import ollama
from pydantic import BaseModel

class EdgeSchema(BaseModel):
    causal: bool
    probability: float
    explanation: str

class llm:
    def __init__(self, model="llama3.2"): 
        self.model = model
        # Configure Ollama client to use the host from environment variable
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.client = ollama.Client(host=ollama_host)
        
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


    def reformat_query(self, query, history):
        sys_prompt = """
        You are a Query Contextualizer.
        Your goal is to rewrite a "Follow-up Query" to be fully self-contained, by incorporating necessary context (Domain, Entity, Topic) from the "Conversation History".

        INSTRUCTIONS:
        1. Read the Conversation History to identify the active subject (e.g., specific domain, ongoing issue, or entities discussed).
        2. Rewrite the Follow-up Query to explicitly include this subject so it makes sense in isolation.
        3. DO NOT change the intent of the question, only narrow its scope.
        4. If the query is already specific, return it unchanged.
        5. Output ONLY the rewritten query text. No explanations.
        """

        # Format history
        history_str = ""
        for turn in history:
            role = "User" if turn["role"] == "user" else "Analyst"
            history_str += f"{role}: {turn['content']}\n"

        curr_prompt = f"""
        --- CONVERSATION HISTORY ---
        {history_str}

        --- FOLLOW-UP QUERY ---
        {query}

        REWRITTEN QUERY:"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": curr_prompt}
            ],
            options={ "num_gpu": 99, "num_ctx": 4096 },
            keep_alive="5m"
        )
        
        return response['message']['content'].strip()


    def query_splitter(self,query):
        prompt = """You are an expert NLU (Natural Language Understanding) router. Your goal is to map a user's input query to the most relevant (Domain, Intent) pairs from a provided list and optimize the query for search.

### Instructions:
1. **Analyze:** Read the user's input query.
2. **Select:** Compare the input against the "Allowed Domain-Intent List" provided below. Select the best matches (maximum 3, sorted by relevance). If the query is unambiguous, select only 1. If no match is found, return an empty 'matches' array: [].
3. **Reform:** For each selected pair, generate a 'reformed_query'. This query must strip conversational filler and inject specific technical keywords or terminology relevant to that specific (Domain, Intent) to assist downstream search retrieval.
4. **Output:** Return the result in the specified JSON format only. DO NOT include any introductory or explanatory text outside of the JSON block.

***IMPORTANT RULE***-
your output for the "domain_intent" field MUST be an integer representing the index (1-based) of the intent within its domain as per the "Allowed Domain-Intent List" below.


### Allowed Domain-Intent List (37 Pairs):
[
1.  ("Banking", "Credit Limit Requests"),
2.  ("Banking", "Fee Complaints"),
3.  ("Banking", "Fraud Alerts"),
4.  ("Banking", "Loan Application"),
5.  ("Banking", "Product Comparison"),
6.  ("Banking", "Refund Delays"),
7.  ("Flight", "Cross Brand Mentions"),
8.  ("Flight", "Delay Management"),
9.  ("Flight", "Loyalty Program"),
10.  ("Flight", "Price Sensitivity"),
11. ("Flight", "Refund Policy"),
12.  ("Flight", "Urgency & Stress"),
13.  ("Hotel", "Booking Errors"),
14.  ("Hotel", "Brand Loyalty"),
15.  ("Hotel", "Cancellation Policies"),
16.  ("Hotel", "Discounts & Promotions"),
17.  ("Hotel", "Service Complaints"),
18.  ("Hotel", "Upgrade Requests"),
19.  ("Insurance", "Claims & Refunds"),
20.  ("Insurance", "Competitor Comparison"),
21.  ("Insurance", "Customer Trust"),
22.  ("Insurance", "Feature Understanding"),
23.  ("Insurance", "Policy Renewal"),
24.  ("Insurance", "Sales Effectiveness"),
25.  ("Insurance", "Upselling Strategy"),
26.  ("Retail", "Delivery Delays"),
27.  ("Retail", "Loyalty Program"),
28.  ("Retail", "Product Feedback"),
29.  ("Retail", "Product Returns"),
30.  ("Retail", "Replacement Vs Refund"),
31.  ("Telecom", "Churn Prediction"),
32.  ("Telecom", "Connectivity Complaints"),
33.  ("Telecom", "Feature Requests"),
34.  ("Telecom", "Network Outages"),
35.  ("Telecom", "Plan Upgrades"),
36.  ("Telecom", "Technical Support")
]

### Output Format (JSON):
{
  "matches": [
    {
      "domain_intent": int,
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
        
        response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
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
            response = self.client.chat(
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
            print(f"Batch inference failed: {e} !!")
            return []

    def answer_query_causal(self, query, chains):
        # IMPROVED PROMPT for follow_up = 0
        sys_prompt = f'''
        You are an expert Root Cause Analysis AI. 
        Your task is to provide a professional, executive-level summary answering the user's query based on the provided causal analysis.

        INPUT DATA:
        - You are given "Causal Chains" containing nodes (metadata like empathy, escalation risk) and edges (probabilities, explanations).
        - The chains are sorted by confidence (cumulative probability).

        STRICT OUTPUT GUIDELINES:
        1. **Narrative Focus**: Synthesize the information into a single, cohesive argument. 
        2. **Hide Mechanics**: DO NOT explicitly mention "Pathway 1", "Chain 3", "Confidence: 0.8", or internal IDs in your final text. The user should not know how the data was structured.
        3. **Analytical Tone**: Instead of saying "Pathway 1 says X", say "Primary analysis indicates X..." or "There is strong evidence that X...".
        4. **Handling Uncertainty**: Use high-probability chains for your main conclusion. Mention lower-probability chains only if they offer distinct, valuable nuance, introducing them with phrases like "Additionally, it is worth noting..." or "A secondary factor appears to be...".
        5. **Metadata Integration**: Weave metadata (e.g., "high churn risk", "low empathy") naturally into your sentences (e.g., "The lack of empathy shown by agents correlates with the high churn risk observed...").
        
        Goal: A direct, seamless answer that sounds like a human expert analyst, not a machine debugging a graph.
        '''

        curr_prompt = f'''Query: {query}
        
        EVIDENCE (CAUSAL CHAINS): 
        {chains}
        
        Provide your expert analysis now:'''

        response = self.client.chat(
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

    def answer_query_causal(self, query, chains):
        # IMPROVED PROMPT for follow_up = 0
        sys_prompt = f'''
        You are an expert Root Cause Analysis AI. 
        Your task is to provide a professional, executive-level summary answering the user's query based on the provided causal analysis.

        INPUT DATA:
        - "Causal Chains": Nodes (metadata) and Edges (probabilities). Sorted by confidence.

        STRICT OUTPUT GUIDELINES:
        1. **STRICT FACTUALITY**: STRICTLY FOLLOW the facts received from the graph (Causal Chains). Do analysis ONLY on them. Do not hallucinate or add outside information.
        2. **Narrative Focus**: Synthesize info into a cohesive argument. Use paragraphs, NOT lists.
        3. **Hide Mechanics**: DO NOT mention "Pathway 1", "Confidence: 0.8", or IDs.
        4. **Analytical Tone**: Say "Primary analysis indicates..." or "Evidence suggests...".
        5. **Metadata Integration**: Weave metadata (e.g., "high churn risk") naturally into sentences.
        6. **Formatting**: Keep output COMPACT. Single spacing between sentences. No multiple blank lines.
        '''

        curr_prompt = f'''Query: {query}
        
        EVIDENCE (CAUSAL CHAINS): 
        {chains}
        
        Provide your expert analysis now:'''

        response = self.client.chat(
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
        # Remove excessive newlines
        return ans.strip().replace("\n\n\n", "\n\n")

    def answer_followup_causal(self, query, chains, history):
        # NEW PROMPT for follow_up > 0
        sys_prompt = f'''
        You are an intelligent Conversational Analyst.
        
        TASK: 
        Answer the "Current Query" using the "New Evidence". 
        Use "Conversation History" ONLY to resolve ambiguous references (e.g., "it", "they", "that issue").

         STRICT OUTPUT GUIDELINES:
        1. **STRICT FACTUALITY**: STRICTLY FOLLOW the facts received from the graph (Causal Chains)  and the previous conversation's context. Do analysis ONLY on them. Do not hallucinate or add outside information.
        2. **Narrative Focus**: Synthesize info into a cohesive argument. Use paragraphs, NOT lists.
        3. **Hide Mechanics**: DO NOT mention "Pathway 1", "Confidence: 0.8", or IDs.
        4. **Analytical Tone**: Say "Primary analysis indicates..." or "Evidence suggests...".
        5. **Metadata Integration**: Weave metadata (e.g., "high churn risk") naturally into sentences.
        6. **Formatting**: Keep output COMPACT. Single spacing between sentences. No multiple blank lines.
        7. **Source of Truth**: The "New Evidence" and "Previous Contexts" are the ONLY factual source for this answer. Ignore factual contradictions from old history.
        8. **Conciseness**: Get straight to the point.
        '''

        # Format history for the prompt
        history_str = ""
        for turn in history:
            role = "User" if turn["role"] == "user" else "Analyst"
            history_str += f"{role}: {turn['content']}\n"

        curr_prompt = f'''
        --- CONVERSATION HISTORY ---
        {history_str}
        
        --- NEW EVIDENCE (CAUSAL CHAINS) ---
        {chains}
        
        --- CURRENT QUERY ---
        {query}
        
        Answer:'''

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": curr_prompt}
            ],
            options={
                "num_gpu": 99,
                "num_ctx": 12000 
            },
            keep_alive="5m"
        )

        ans = response['message']['content']
        # Remove excessive newlines
        return ans.strip().replace("\n\n\n", "\n\n")