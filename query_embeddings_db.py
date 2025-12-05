#!/usr/bin/env python3
"""
Query-only pipeline that loads pre-computed embeddings from database.
This is used by graph_gen.py to avoid re-computing embeddings.
"""

import json
import sqlite3
import numpy as np
import pickle
import faiss
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class CachedEmbeddingSearch:
    """Search using cached embeddings from database."""
    
    def __init__(self, db_path: str, data_json_path: str = None, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize search with cached embeddings.
        
        Args:
            db_path: Path to embeddings database
            data_json_path: Path to original JSON data (to load full conversations)
            model_name: Name of sentence transformer model
        """
        self.db_path = db_path
        self.data_json_path = data_json_path
        self.faiss_index_path = db_path.replace('.db', '.faiss')
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Load original data if provided
        self.transcript_data = {}
        if data_json_path:
            print(f"Loading original data from: {data_json_path}")
            with open(data_json_path, 'r') as f:
                data = json.load(f)
            for item in data:
                self.transcript_data[item['transcript_id']] = item
        
        # Load FAISS index
        print(f"Loading FAISS index from: {self.faiss_index_path}")
        self.index = faiss.read_index(self.faiss_index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
    def search_by_domain_and_query(
        self,
        domain: Optional[str],
        query: str,
        top_k: int = 20,
        intent: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for top-k transcripts in a domain matching the query.
        
        Args:
            domain: Domain to filter by
            query: Search query
            top_k: Number of top transcripts to return
            
        Returns:
            List of top transcript results with metadata
        """
        print(f"\n{'='*80}")
        print(f"SEARCH QUERY")
        print(f"{'='*80}")
        print(f"Domain: {domain or 'All'}")
        print(f"Intent filter: {intent if intent else 'any'}")
        print(f"Query: {query}")
        print(f"Top K: {top_k}")
        print(f"{'='*80}\n")
        
        # Load embeddings for this domain from database
        load_scope = f"domain '{domain}'" if domain else "all domains"
        print(f"Loading embeddings for {load_scope}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        base_query = '''
            SELECT id, transcript_id, domain, intent, reason_for_call, turn_idx,
                   speaker, text, combined_text, embedding,
                   escalation_level, churn_risk_score, empathy_score,
                   intents_emotions, dialogue_acts, action_type,
                   escalation_reason_tags, time_of_interaction
            FROM embeddings
        '''

        conditions = []
        query_params: List[str] = []
        if domain:
            conditions.append('domain = ?')
            query_params.append(domain)
        if intent:
            conditions.append('LOWER(intent) = LOWER(?)')
            query_params.append(intent)

        if conditions:
            base_query += ' WHERE ' + ' AND '.join(conditions)

        base_query += ' ORDER BY id'

        cursor.execute(base_query, tuple(query_params))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            scope_parts = []
            if domain:
                scope_parts.append(f"domain '{domain}'")
            if intent:
                scope_parts.append(f"intent '{intent}'")
            scope = ' and '.join(scope_parts) if scope_parts else 'database'
            print(f"No embeddings found for {scope}")
            return []
        
        detail_parts = []
        if domain:
            detail_parts.append(f"domain {domain}")
        if intent:
            detail_parts.append(f"intent {intent}")
        detail_str = ' and '.join(detail_parts) if detail_parts else 'all data'
        print(f"Loaded {len(rows)} turns for {detail_str}")
        
        # Extract embeddings and metadata
        domain_embeddings = []
        turn_metadata = []
        
        for row in rows:
            (id_, transcript_id, domain, intent, reason_for_call, turn_idx,
             speaker, text, combined_text, embedding_blob,
             escalation_level, churn_risk_score, empathy_score,
             intents_emotions, dialogue_acts, action_type,
             escalation_reason_tags, time_of_interaction) = row
            
            embedding = pickle.loads(embedding_blob)
            domain_embeddings.append(embedding)
            
            turn_metadata.append({
                'db_id': id_,
                'transcript_id': transcript_id,
                'domain': domain,
                'intent': intent,
                'reason_for_call': reason_for_call,
                'turn_idx': turn_idx,
                'speaker': speaker,
                'text': text,
                'combined_text': combined_text,
                'escalation_level': escalation_level,
                'churn_risk_score': churn_risk_score,
                'empathy_score': empathy_score,
                'intents_emotions': intents_emotions,
                'dialogue_acts': dialogue_acts,
                'action_type': action_type,
                'escalation_reason_tags': json.loads(escalation_reason_tags) if escalation_reason_tags else [],
                'time_of_interaction': time_of_interaction
            })
        
        domain_embeddings = np.array(domain_embeddings)
        
        # Build temporary FAISS index for this domain
        print("Building domain-specific FAISS index...")
        faiss.normalize_L2(domain_embeddings)
        
        embedding_dim = domain_embeddings.shape[1]
        domain_index = faiss.IndexFlatIP(embedding_dim)
        domain_index.add(domain_embeddings.astype('float32'))
        
        # Encode query
        print(f"\nSearching for top turns matching query: '{query}'")
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search - get more results initially for aggregation
        k_turns = min(200, len(domain_embeddings))
        distances, indices = domain_index.search(query_embedding.astype('float32'), k_turns)
        
        # Aggregate by transcript
        print("\nAggregating scores to transcript level...")
        transcript_scores = {}
        
        for dist, idx in zip(distances[0], indices[0]):
            metadata = turn_metadata[idx]
            transcript_id = metadata['transcript_id']
            
            if transcript_id not in transcript_scores:
                transcript_scores[transcript_id] = {
                    'transcript_id': transcript_id,
                    'domain': metadata['domain'],
                    'intent': metadata['intent'],
                    'reason_for_call': metadata['reason_for_call'],
                    'time_of_interaction': metadata['time_of_interaction'],
                    'max_similarity': float(dist),
                    'total_similarity': 0.0,
                    'num_turns': 0,
                    'matching_turns': []
                }
            
            transcript_scores[transcript_id]['total_similarity'] += float(dist)
            transcript_scores[transcript_id]['num_turns'] += 1
            transcript_scores[transcript_id]['matching_turns'].append({
                'turn_idx': metadata['turn_idx'],
                'speaker': metadata['speaker'],
                'text': metadata['text'],
                'similarity_score': float(dist)
            })
            
            # Update max if needed
            if dist > transcript_scores[transcript_id]['max_similarity']:
                transcript_scores[transcript_id]['max_similarity'] = float(dist)
        
        # Calculate average and sort
        for transcript_id in transcript_scores:
            num_turns = transcript_scores[transcript_id]['num_turns']
            transcript_scores[transcript_id]['avg_similarity'] = (
                transcript_scores[transcript_id]['total_similarity'] / num_turns
            )
            # Sort matching turns by similarity
            transcript_scores[transcript_id]['matching_turns'].sort(
                key=lambda x: x['similarity_score'], reverse=True
            )
        
        # Sort by max similarity
        sorted_transcripts = sorted(
            transcript_scores.values(),
            key=lambda x: x['max_similarity'],
            reverse=True
        )[:top_k]
        
        # Add full conversation data and metadata for each result
        for result in sorted_transcripts:
            result['similarity_metadata'] = {
                'max_similarity_score': result['max_similarity'],
                'avg_similarity_score': result['avg_similarity'],
                'num_matching_turns': result['num_turns'],
                'top_matching_turns': result['matching_turns'][:5]
            }
            
            # Add full conversation from original data
            transcript_id = result['transcript_id']
            if self.transcript_data and transcript_id in self.transcript_data:
                original_data = self.transcript_data[transcript_id]
                result['conversation'] = original_data.get('conversation', [])
                result['satisfaction_score'] = original_data.get('satisfaction_score')
                result['satisfaction_label'] = original_data.get('satisfaction_label')
        
        print(f"Found {len(transcript_scores)} unique transcripts, returning top {len(sorted_transcripts)}")
        
        return sorted_transcripts

def run_cached_search(
        db_path: str,
        domain: Optional[str],
        query: str,
        top_k: int = 20,
        intent: Optional[str] = None
    ) -> List[Dict]:
    """Convenience wrapper for cached search."""
    searcher = CachedEmbeddingSearch(db_path)
    return searcher.search_by_domain_and_query(domain, query, top_k, intent)
