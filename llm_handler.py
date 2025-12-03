import os
from xmlrpc.client import boolean
from groq import Groq
import json
import ollama
from pydantic import BaseModel
ct = 0
class EdgeSchema(BaseModel):
    causal: bool
    probability: float
    explanation: str

class llm:
    def __init__(self, model="llama3.1"): 
        self.model = model
        self.base_prompt = """
You are an expert in causal reasoning for conversational analysis. 
Your task is to determine if the 'Source Node' is the CAUSE of the 'Target Node's' content AND OUTPUT ONLY A JSON FILE.
Each node has the following metadata :
    1. intent_emotions depicting the intents and emotions detected in the utterance.
    2. escalation_level indicating how escalated the conversation is because of this utterance.
    3. churn_risk_score indicating the risk of customer churn because of this utterance
    4. empathy_score indicating the level of empathy shown in this utterance.
    5. escalation_risks indicating the risk factors contributing to escalation because of this utterance.
    6. speaker indicating who is speaking in this utterance.

USE THE ABOVE METADATA CAREFULLY IN YOUR ANALYSIS.

INPUT DATA:
You will receive two JSON objects containing text and metadata. YOU NEED TO OUTPUT ONLY A JSON FILE

TASK:
1. Analyze if the event/statement in the SOURCE directly influenced the TARGET.
2. Also, look for reasons explaining why the TARGET node has the given scores which are in metadata.
3. Pay close attention to the text which is stored in utterances field of both nodes.

SCORING GUIDE:
- 0.0: No causal link, or Target is not negative/reactive.
- 0.1 - 0.5: Weak link (topic match, but loose causality).
- 0.6 - 1.0: Strong causal link (direct response, complaint about specific entity mentioned in Source).

OUTPUT FORMAT:
Return STRICT JSON only. Do not use Markdown codes (no ```json). 
Format:
{
    "causal": boolean,
    "probability": float,
    "explanation": "concise reason referencing specific entities/topics"
}
"""

    def generate_edge(self,node_1,node_2):
        #print(node_1)


        prompt = f'''Here is the data of both the nodes
            Node1 --
            {node_1}

            Node2 --
            {node_2}
        '''

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ],
            format="json"
        )

        try:
            ans = json.loads(response['message']['content'])
            return ans
        except json.JSONDecodeError:
            print("Error decoding JSON from Ollama response")
            return {
                "causal": False,
                "probability": 0.0,
                "explanation": "Failed to parse LLM response"
            }
    
    def answer_query(self,query,context):
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
            ]
        )

        ans = response['message']['content']
        #print(ans)
        return ans

    def answer_query_causal(self,query,chains):
        sys_prompt = f'''
        You are an expert Root Cause Analysis AI.
        Your task is to perform causal analysis given the causal chains related to the query and a query.
        You will be given a causal chain in the form of NODE , EDGE , NODE ... format. For each node, you will have metadata such as utterance , empathy score, escalation risk, churn risk , intent_emotions etc. For each edge, you will have an explanation and a cumulative probability score.
        The edges represent the causal relationships between the nodes.
        Your goal is to analyze these chains to provide a concise answer to the user's query.
        


        INSTRUCTIONS:
        1. REVIEW: Read the provided chains (ordered: Cause -> Effect).
        2.ANALYZE METADATA : Analyze all the metadata such as utterance , empathy score, escalation risk, churn risk , intent_emotions etc in the nodes of the chains to get better context on the causal relationships.
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
            ]
        )

        ans = response['message']['content']
        #print(ans)
        return ans