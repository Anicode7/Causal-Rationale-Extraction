import os
import json
import logging
import llm_handler
import find_top_20_conversations
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

def categorize_query(query, llm, embeddings_db, data_path):
	"""
	Categorizes the query based on the provided domain and intent.
	Returns a dictionary with the query, domain, and intent.
	"""


	searcher = query_embeddings_db.CachedEmbeddingSearch(embeddings_db, data_path)

	queries_list = llm.query_splitter(query)
	unique_conversations = {}
	num_trs = 20
	conversation_search_pipeline_obj = find_top_20_conversations.ConversationSearchPipeline()
	for i, data in enumerate(queries_list):
		curr_domain = data["domain"]
		curr_query = data["reformed_query"]
		curr_intent = data.get("intent")

		print(f"Processing query {i+1}/{len(queries_list)}: Domain: {curr_domain}, Intent: {curr_intent}, Query: {curr_query}")
		curr_transcripts = searcher.search_by_domain_and_query(domain=curr_domain, query=curr_query, intent=curr_intent, top_k=num_trs)
		for t in curr_transcripts:
			tid = t.get('transcript_id')
			if tid:
				unique_conversations[tid] = t

		if curr_transcripts:
			top20_dir = os.path.join(os.getcwd(), 'data/top_20')
			os.makedirs(top20_dir, exist_ok=True)

			# Save summary JSON
			from datetime import datetime
			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			domain_str = sanitize_token(curr_domain)
			intent_str = sanitize_token(curr_intent, default='any')
			summary_file = os.path.join(
				top20_dir,
				f'top_20_{domain_str}_{intent_str}_{timestamp}.json'
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
				txt_file = os.path.join(top20_dir, f'{transcript_id}.txt')

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

			logger.info(f"✓ Found {len(results)} relevant conversations")
			logger.info(f"✓ Saved to {top20_dir}\n")

	return list(unique_conversations.values())








