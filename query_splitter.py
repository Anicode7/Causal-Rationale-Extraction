import os
import json
import llm_handler

def categorize_query(query,llm):
    """
    Categorizes the query based on the provided domain and intent.
    Returns a dictionary with the query, domain, and intent.
    """
    queries_list = llm.query_splitter(query)
    num_trs = 20







