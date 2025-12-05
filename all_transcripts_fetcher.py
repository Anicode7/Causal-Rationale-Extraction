import os
import json
import glob
import logging
# ...existing code...

logger = logging.getLogger(__name__)

def sanitize_token(token, default='any'):
	"""Keep filename-safe chars for top_K filenames."""
	if not token:
		return default
	return "".join(c for c in str(token) if c.isalnum() or c in ('-', '_')).lower()

def _load_top20_from_dir(top20_dir):
	"""
	Load saved top_K_*.json files from top20_dir and return a dict of transcript_id -> transcript_obj.
	"""
	found = {}
	if not os.path.isdir(top20_dir):
		return found
	for path in glob.glob(os.path.join(top20_dir, "top_K_*.json")):
		try:
			with open(path, "r") as fh:
				data = json.load(fh)
				for r in data.get("results", []):
					tid = r.get("transcript_id")
					if tid:
						found[tid] = r
		except Exception:
			logger.exception("Failed loading top20 file: %s", path)
	return found

def categorize_query(query, llm, embeddings_db, data_path):
	# ...existing code...
	searcher =  query_embeddings_db.CachedEmbeddingSearch(embeddings_db, data_path)

	queries_list = llm.query_splitter(query)
	unique_conversations = {}
	num_trs = 20
	conversation_search_pipeline_obj = find_top_K_conversations.ConversationSearchPipeline()
	for i, data in enumerate(queries_list):
		# ...existing code...
		curr_domain = data["domain"]
		curr_query = data["reformed_query"]
		curr_intent = data.get("intent")
		# ...existing code...
		curr_transcripts = searcher.search_by_domain_and_query(domain=curr_domain, query=curr_query, intent=curr_intent, top_k=num_trs)
		for t in curr_transcripts:
			tid = t.get('transcript_id')
			if tid:
				unique_conversations[tid] = t
		# ...existing code that writes top_K files ...
	
	# Merge any transcripts already saved in data/top_K so dialogue flow can use all of them
	top20_dir = os.path.join(os.getcwd(), 'data', 'top_K')
	saved = _load_top20_from_dir(top20_dir)
	if saved:
		for tid, obj in saved.items():
			if tid not in unique_conversations:
				unique_conversations[tid] = obj
		logger.info("Merged %d transcripts from %s into results", len(saved), top20_dir)

	return list(unique_conversations.values())