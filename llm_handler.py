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

        response = ollama.chat(
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

        response = ollama.chat(
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