import os
import json
import logging
import llm_handler
import query_embeddings_db

# add logger and helper
logger = logging.getLogger(__name__)

def sanitize_token(token, default='any'):
	"""
	Simple sanitizer for tokens used in filenames.
	"""
	if not token:
		return default
	# keep only safe filename characters
	return "".join(c for c in token if c.isalnum() or c in ('-', '_')).lower()

def save_list(filename, data_list):
    with open(filename, "w") as f:
        for item in data_list:
            f.write(str(item) + "\n")

def load_list(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f]

def categorize_query(query, llm, embeddings_db, data_path,follow_up,topK_dir,topk):
	"""
	Categorizes the query based on the provided domain and intent.
	Returns a dictionary with the query, domain, and intent.
	"""

	searcher = query_embeddings_db.CachedEmbeddingSearch(embeddings_db, data_path)
	queries_list_old = llm.query_splitter(query)
	list_intent = [
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
    ("Telecom", "Technical Support"),
]
	queries_list = []
	print("hiiiiiii" , queries_list_old)
	for obj in queries_list_old:
		id = obj["domain_intent"]
		curr_domain = list_intent[id-1][0]
		curr_intent = list_intent[id-1][1]
		curr_query = obj["reformed_query"]
		reas = obj["reasoning"]
		queries_list.append({
			"domain": curr_domain,
			"reformed_query": curr_query,
			"intent": curr_intent,
			"reasoning": reas
		})
	

	print("byee " , queries_list)
	unique_conversations = {}
	transcript_id_list = []
	if(follow_up!=0):
		transcript_id_list = load_list(os.path.join("data", "transcript_id_list.txt"))
	transcript_id_dict = {tid: True for tid in transcript_id_list}
	for i, data in enumerate(queries_list):
		curr_domain = data["domain"]
		curr_query = data["reformed_query"]
		curr_intent = data.get("intent")

		print(f"Processing query {i+1}/{len(queries_list)}: Domain: {curr_domain}, Intent: {curr_intent}, Query: {curr_query}")
		curr_transcripts = searcher.search_by_domain_and_query(domain=curr_domain, query=curr_query, intent=curr_intent, top_k=topk)
		for t in curr_transcripts:
			tid = t.get('transcript_id')
			if(tid not in transcript_id_dict):
				transcript_id_list.append(tid)
				transcript_id_dict[tid] = True
			if tid:
				unique_conversations[tid] = t

		if curr_transcripts:
			os.makedirs(topK_dir, exist_ok=True)

			# Save summary JSON
			from datetime import datetime
			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			domain_str = sanitize_token(curr_domain)
			intent_str = sanitize_token(curr_intent, default='any')
			summary_file = os.path.join(
				topK_dir,
				f'top_K_{domain_str}_{intent_str}_{timestamp}.json'
			)

			# Use current transcripts as results
			results = curr_transcripts

			# Wrap results in the expected format
			output_data = {
				'query': query,
				'domain': curr_domain,
				'intent': curr_intent,
				'timestamp': timestamp,
				'total_results': len(results),
				'results': results
			}

			with open(summary_file, 'w') as f:
				json.dump(output_data, f, indent=2)

			# Save individual transcript text files
			for j, result in enumerate(results, 1):
				transcript_id = result.get('transcript_id', f'unknown_{j}')
				txt_file = os.path.join(topK_dir, f'{transcript_id}.txt')

				with open(txt_file, 'w') as f:
					f.write(f"Transcript ID: {transcript_id}\n")
					f.write(f"Domain: {result.get('domain')}\n")
					f.write(f"Intent: {result.get('intent')}\n")
					f.write(f"Reason: {result.get('reason_for_call')}\n")

					# similarity metadata may not exist; guard access
					sim_meta = result.get('similarity_metadata', {})
					max_sim = sim_meta.get('max_similarity_score', 0.0)
					avg_sim = sim_meta.get('avg_similarity_score', 0.0)
					f.write(f"Max Similarity: {max_sim:.4f}\n")
					f.write(f"Avg Similarity: {avg_sim:.4f}\n")
					f.write("\nTop Matching Turns:\n")
					f.write("=" * 60 + "\n")
					for turn in sim_meta.get('top_matching_turns', [])[:5]:
						f.write(f"\n{turn.get('speaker')}: {turn.get('text')}\n")
						f.write(f"Similarity: {turn.get('similarity_score', 0.0):.4f}\n")

			logger.info(f"Found {len(results)} relevant conversations")
			logger.info(f"Saved to {topK_dir}\n")
	# Save updated transcript ID list if follow_up is enabled
	save_list(os.path.join("data", "transcript_id_list.txt"), transcript_id_list)
	return list(unique_conversations.values()), queries_list








