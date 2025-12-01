#!/usr/bin/env python3
"""
Pipeline to find the top 20 most relevant conversations based on semantic similarity.

This script:
1. Loads conversation data from JSON
2. Filters by domain
3. Creates embeddings for each turn with metadata
4. Performs similarity search using Faiss
5. Returns top 20 most relevant transcripts
"""

import json
import os
import numpy as np
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
from datetime import datetime


class ConversationSearchPipeline:
    """Pipeline for finding relevant conversations using semantic similarity."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize the pipeline.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.data = None
        self.embeddings = None
        self.index = None
        self.turn_metadata = []  # Store metadata for each turn
        
    def load_data(self, json_file_path: str) -> List[Dict]:
        """
        Load conversation data from JSON file.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            List of conversation transcripts
        """
        print(f"Loading data from: {json_file_path}")
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        print(f"Loaded {len(data)} transcripts")
        return data
    
    def filter_by_domain(self, data: List[Dict], domain: str) -> List[Dict]:
        """
        Filter transcripts by domain.
        
        Args:
            data: List of all transcripts
            domain: Domain to filter by (e.g., 'Flight', 'Hotel', 'Banking')
            
        Returns:
            Filtered list of transcripts for the specified domain
        """
        filtered = [t for t in data if t.get('domain', '').lower() == domain.lower()]
        print(f"Filtered to {len(filtered)} transcripts for domain: {domain}")
        return filtered
    
    def prepare_turn_text(self, turn: Dict, reason_for_call: str = "") -> str:
        """
        Combine turn text with metadata for embedding.
        
        Args:
            turn: Dictionary containing turn information
            reason_for_call: Reason for call from transcript level
            
        Returns:
            Combined text string
        """
        # Extract metadata fields
        text = turn.get('text', '')
        intents_emotions = turn.get('intents_emotions', '')
        dialogue_acts = turn.get('dialogue_acts', '')
        action_type = turn.get('action_type', '')
        escalation_tags = ' '.join(turn.get('escalation_reason_tags', []))
        
        # Combine all information
        combined = f"{text} | Intent/Emotion: {intents_emotions} | Dialogue Act: {dialogue_acts} | Action: {action_type} | Tags: {escalation_tags}"
        
        # Add reason for call if provided
        if reason_for_call:
            combined = f"Reason: {reason_for_call} | {combined}"
        
        return combined
    
    def create_embeddings(self, data: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Create embeddings for all turns in the dataset.
        
        Args:
            data: List of transcripts
            
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        print("Creating embeddings for all conversation turns...")
        all_texts = []
        turn_metadata = []
        
        for transcript_idx, transcript in enumerate(data):
            reason_for_call = transcript.get('reason_for_call', '')
            transcript_id = transcript.get('transcript_id', '')
            domain = transcript.get('domain', '')
            intent = transcript.get('intent', '')
            time_of_interaction = transcript.get('time_of_interaction', '')
            
            conversation = transcript.get('conversation', [])
            
            for turn_idx, turn in enumerate(conversation):
                # Prepare combined text
                combined_text = self.prepare_turn_text(turn, reason_for_call)
                all_texts.append(combined_text)
                
                # Store metadata
                turn_metadata.append({
                    'transcript_idx': transcript_idx,
                    'transcript_id': transcript_id,
                    'domain': domain,
                    'intent': intent,
                    'time_of_interaction': time_of_interaction,
                    'reason_for_call': reason_for_call,
                    'turn_idx': turn_idx,
                    'speaker': turn.get('speaker', ''),
                    'text': turn.get('text', ''),
                    'escalation_level': turn.get('escalation_level', 0),
                    'churn_risk_score': turn.get('churn_risk_score', 0),
                    'empathy_score': turn.get('empathy_score', 0),
                    'intents_emotions': turn.get('intents_emotions', ''),
                    'dialogue_acts': turn.get('dialogue_acts', ''),
                    'action_type': turn.get('action_type', ''),
                    'escalation_reason_tags': turn.get('escalation_reason_tags', [])
                })
        
        print(f"Encoding {len(all_texts)} conversation turns...")
        embeddings = self.model.encode(all_texts, show_progress_bar=True)
        print(f"Created embeddings with shape: {embeddings.shape}")
        
        return embeddings, turn_metadata
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a Faiss index for efficient similarity search.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Faiss index
        """
        print("Building Faiss index...")
        embedding_dim = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (inner product)
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        print(f"Faiss index built with {index.ntotal} vectors")
        return index
    
    def search_similar_turns(self, query: str, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar turns using the query.
        
        Args:
            query: Query string
            k: Number of results to return (we get more initially for transcript-level aggregation)
            
        Returns:
            Tuple of (distances, indices)
        """
        print(f"\nSearching for top turns matching query: '{query}'")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        return distances[0], indices[0]
    
    def aggregate_transcript_scores(self, distances: np.ndarray, indices: np.ndarray, top_k: int = 20) -> List[Dict]:
        """
        Aggregate turn-level scores to transcript-level and get top K transcripts.
        
        Args:
            distances: Similarity scores for each turn
            indices: Indices of similar turns
            top_k: Number of top transcripts to return
            
        Returns:
            List of top K transcripts with metadata
        """
        print(f"\nAggregating scores to transcript level...")
        
        # Dictionary to store transcript scores
        transcript_scores = {}
        
        for dist, idx in zip(distances, indices):
            metadata = self.turn_metadata[idx]
            transcript_id = metadata['transcript_id']
            
            if transcript_id not in transcript_scores:
                transcript_scores[transcript_id] = {
                    'transcript_id': transcript_id,
                    'transcript_idx': metadata['transcript_idx'],
                    'domain': metadata['domain'],
                    'intent': metadata['intent'],
                    'time_of_interaction': metadata['time_of_interaction'],
                    'reason_for_call': metadata['reason_for_call'],
                    'max_similarity': dist,
                    'avg_similarity': dist,
                    'similarity_scores': [dist],
                    'matching_turns': [metadata],
                    'num_matching_turns': 1
                }
            else:
                # Update scores
                transcript_scores[transcript_id]['similarity_scores'].append(dist)
                transcript_scores[transcript_id]['matching_turns'].append(metadata)
                transcript_scores[transcript_id]['num_matching_turns'] += 1
                transcript_scores[transcript_id]['max_similarity'] = max(
                    transcript_scores[transcript_id]['max_similarity'], dist
                )
                transcript_scores[transcript_id]['avg_similarity'] = np.mean(
                    transcript_scores[transcript_id]['similarity_scores']
                )
        
        # Sort by max similarity (you can also use avg_similarity or a weighted combination)
        sorted_transcripts = sorted(
            transcript_scores.values(),
            key=lambda x: (x['max_similarity'], x['avg_similarity'], x['num_matching_turns']),
            reverse=True
        )
        
        # Get top K
        top_transcripts = sorted_transcripts[:top_k]
        
        print(f"Found {len(sorted_transcripts)} unique transcripts, returning top {len(top_transcripts)}")
        
        return top_transcripts
    
    def get_full_transcripts(self, top_results: List[Dict]) -> List[Dict]:
        """
        Get full transcript data for the top results.
        
        Args:
            top_results: List of top transcript metadata
            
        Returns:
            List of full transcripts with similarity metadata
        """
        full_transcripts = []
        
        for result in top_results:
            transcript_idx = result['transcript_idx']
            full_transcript = self.data[transcript_idx].copy()
            
            # Add similarity metadata
            full_transcript['similarity_metadata'] = {
                'max_similarity_score': float(result['max_similarity']),
                'avg_similarity_score': float(result['avg_similarity']),
                'num_matching_turns': result['num_matching_turns'],
                'top_matching_turns': [
                    {
                        'turn_idx': turn['turn_idx'],
                        'speaker': turn['speaker'],
                        'text': turn['text'],
                        'similarity_score': float(score)
                    }
                    for turn, score in sorted(
                        zip(result['matching_turns'], result['similarity_scores']),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]  # Top 5 matching turns
                ]
            }
            
            full_transcripts.append(full_transcript)
        
        return full_transcripts
    
    def save_results(self, results: List[Dict], output_dir: str, domain: str, query: str):
        """
        Save the top results as individual text files and a summary JSON.
        
        Args:
            results: List of top transcripts
            output_dir: Directory to save results
            domain: Domain name
            query: Query string
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_domain = domain.replace(' ', '_').lower()
        
        # Save summary JSON
        json_filename = f"top_20_{safe_domain}_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        output_data = {
            'query': query,
            'domain': domain,
            'timestamp': timestamp,
            'num_results': len(results),
            'results': results
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Summary JSON saved to: {json_path}")
        
        # Save each transcript as individual text file
        print(f"✓ Saving {len(results)} individual transcripts as text files...")
        
        for idx, result in enumerate(results, 1):
            transcript_id = result['transcript_id']
            txt_filename = f"{transcript_id}.txt"
            txt_path = os.path.join(output_dir, txt_filename)
            
            # Format the transcript as readable text with ALL original data
            with open(txt_path, 'w') as f:
                # Header
                f.write("="*80 + "\n")
                f.write(f"TRANSCRIPT ID: {transcript_id}\n")
                f.write("="*80 + "\n\n")
                
                # Transcript-level metadata (all fields from original JSON)
                f.write("-"*80 + "\n")
                f.write("TRANSCRIPT METADATA\n")
                f.write("-"*80 + "\n")
                f.write(f"Transcript ID: {result.get('transcript_id', 'N/A')}\n")
                f.write(f"Domain: {result.get('domain', 'N/A')}\n")
                f.write(f"Intent: {result.get('intent', 'N/A')}\n")
                f.write(f"Time of Interaction: {result.get('time_of_interaction', 'N/A')}\n")
                f.write(f"Reason for Call: {result.get('reason_for_call', 'N/A')}\n")
                
                # Include any other transcript-level fields that might exist
                excluded_keys = {'transcript_id', 'domain', 'intent', 'time_of_interaction', 
                               'reason_for_call', 'conversation', 'similarity_metadata'}
                for key, value in result.items():
                    if key not in excluded_keys:
                        f.write(f"{key}: {value}\n")
                
                f.write("\n")
                
                # Similarity metadata (search-specific, not from original JSON)
                sim_meta = result.get('similarity_metadata', {})
                f.write("-"*80 + "\n")
                f.write("SEARCH SIMILARITY SCORES (Generated by Search)\n")
                f.write("-"*80 + "\n")
                f.write(f"Search Rank: #{idx}\n")
                f.write(f"Max Similarity Score: {sim_meta.get('max_similarity_score', 0):.4f}\n")
                f.write(f"Avg Similarity Score: {sim_meta.get('avg_similarity_score', 0):.4f}\n")
                f.write(f"Number of Matching Turns: {sim_meta.get('num_matching_turns', 0)}\n\n")
                
                # Top matching turns (search-specific)
                top_turns = sim_meta.get('top_matching_turns', [])
                if top_turns:
                    f.write("-"*80 + "\n")
                    f.write("TOP MATCHING TURNS (Generated by Search)\n")
                    f.write("-"*80 + "\n")
                    for i, turn in enumerate(top_turns, 1):
                        f.write(f"\n{i}. Turn #{turn.get('turn_idx', 'N/A')} - {turn.get('speaker', 'N/A')}\n")
                        f.write(f"   Similarity: {turn.get('similarity_score', 0):.4f}\n")
                        f.write(f"   Text: {turn.get('text', 'N/A')}\n")
                    f.write("\n")
                
                # Full conversation with ALL original data
                f.write("="*80 + "\n")
                f.write("FULL CONVERSATION (All Original Data)\n")
                f.write("="*80 + "\n\n")
                
                conversation = result.get('conversation', [])
                for turn_idx, turn in enumerate(conversation):
                    f.write(f"{'='*80}\n")
                    f.write(f"Turn #{turn_idx}\n")
                    f.write(f"{'='*80}\n")
                    
                    # Write ALL fields from the turn
                    f.write(f"Speaker: {turn.get('speaker', 'Unknown')}\n")
                    f.write(f"Text: {turn.get('text', '')}\n\n")
                    
                    # All numeric scores
                    f.write(f"Escalation Level: {turn.get('escalation_level', 'N/A')}\n")
                    f.write(f"Escalation Risks: {turn.get('escalation_risks', 'N/A')}\n")
                    f.write(f"Churn Risk Score: {turn.get('churn_risk_score', 'N/A')}\n")
                    f.write(f"Empathy Score: {turn.get('empathy_score', 'N/A')}\n\n")
                    
                    # All text annotations
                    f.write(f"Intents/Emotions: {turn.get('intents_emotions', 'N/A')}\n")
                    f.write(f"Dialogue Acts: {turn.get('dialogue_acts', 'N/A')}\n")
                    f.write(f"Action Type: {turn.get('action_type', 'N/A')}\n")
                    
                    # Escalation tags
                    escalation_tags = turn.get('escalation_reason_tags', [])
                    if escalation_tags:
                        f.write(f"Escalation Reason Tags: {', '.join(escalation_tags)}\n")
                    else:
                        f.write(f"Escalation Reason Tags: None\n")
                    
                    # Include any additional fields that might exist
                    excluded_turn_keys = {'speaker', 'text', 'escalation_level', 'escalation_risks',
                                         'churn_risk_score', 'empathy_score', 'intents_emotions',
                                         'dialogue_acts', 'action_type', 'escalation_reason_tags'}
                    for key, value in turn.items():
                        if key not in excluded_turn_keys:
                            f.write(f"{key}: {value}\n")
                    
                    f.write("\n")
                
                # Footer
                f.write("="*80 + "\n")
                f.write(f"END OF TRANSCRIPT: {transcript_id}\n")
                f.write("="*80 + "\n")
        
        print(f"✓ All transcripts saved to: {output_dir}/")
        
        return json_path
    
    def run_pipeline(self, json_file_path: str, domain: str, query: str, 
                     output_dir: str = None, top_k: int = 20):
        """
        Run the complete pipeline.
        
        Args:
            json_file_path: Path to input JSON file
            domain: Domain to filter by
            query: Search query
            output_dir: Directory to save results (default: data/top_20/)
            top_k: Number of top results to return
        """
        if output_dir is None:
            # Use data/top_20/ relative to the script location
            script_dir = Path(__file__).parent
            output_dir = script_dir / 'data' / 'top_20'
        
        # Step 1: Load data
        all_data = self.load_data(json_file_path)
        
        # Step 2: Filter by domain
        self.data = self.filter_by_domain(all_data, domain)
        
        if len(self.data) == 0:
            print(f"Error: No transcripts found for domain '{domain}'")
            available_domains = set(t.get('domain', '') for t in all_data)
            print(f"Available domains: {sorted(available_domains)}")
            return
        
        # Step 3: Create embeddings
        self.embeddings, self.turn_metadata = self.create_embeddings(self.data)
        
        # Step 4: Build Faiss index
        self.index = self.build_faiss_index(self.embeddings)
        
        # Step 5: Search for similar turns
        # Get more turns initially for better transcript-level aggregation
        search_k = min(len(self.turn_metadata), 100)
        distances, indices = self.search_similar_turns(query, k=search_k)
        
        # Step 6: Aggregate to transcript level and get top K
        top_results = self.aggregate_transcript_scores(distances, indices, top_k=top_k)
        
        # Step 7: Get full transcripts
        full_transcripts = self.get_full_transcripts(top_results)
        
        # Step 8: Save results
        output_path = self.save_results(full_transcripts, output_dir, domain, query)
        
        # Print summary
        print("\n" + "="*80)
        print("SEARCH SUMMARY")
        print("="*80)
        print(f"Domain: {domain}")
        print(f"Query: {query}")
        print(f"Total transcripts in domain: {len(self.data)}")
        print(f"Top {len(full_transcripts)} results saved to: {output_path}")
        print("\nTop 5 Results:")
        for i, result in enumerate(full_transcripts[:5], 1):
            sim_meta = result['similarity_metadata']
            print(f"\n{i}. Transcript ID: {result['transcript_id']}")
            print(f"   Intent: {result.get('intent', 'N/A')}")
            print(f"   Max Similarity: {sim_meta['max_similarity_score']:.4f}")
            print(f"   Avg Similarity: {sim_meta['avg_similarity_score']:.4f}")
            print(f"   Matching Turns: {sim_meta['num_matching_turns']}")
            print(f"   Reason: {result.get('reason_for_call', 'N/A')[:100]}...")
        
        print("\n" + "="*80)
        
        return full_transcripts


def main():
    """Main function to run the pipeline from command line."""
    parser = argparse.ArgumentParser(
        description='Find top 20 most relevant conversations using semantic similarity'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='/home/pushpendras0026/dialog2flow/data/final_json_for_d2f.json',
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        help='Domain to search within (e.g., Flight, Hotel, Banking, Retail, Telecom, Insurance)'
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Search query'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/pushpendras0026/dialog2flow/data/top_20',
        help='Directory to save results'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of top results to return (default: 20)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Sentence transformer model to use'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ConversationSearchPipeline(model_name=args.model)
    
    # Run pipeline
    pipeline.run_pipeline(
        json_file_path=args.data_path,
        domain=args.domain,
        query=args.query,
        output_dir=args.output_dir,
        top_k=args.top_k
    )


if __name__ == '__main__':
    main()
