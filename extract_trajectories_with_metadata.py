#!/usr/bin/env python3
"""
Extract trajectories from dialogues with metadata propagation.
This is an enhanced version of Dialog2Flow's extract_trajectories.py that:
1. Maintains metadata from conversations_metadata.json
2. Aggregates metadata at cluster level (average numerical, concatenate text)
3. Outputs enhanced trajectories with metadata
4. Generates cluster labels using LLM (ollama/OpenAI)

Based on Dialog2Flow by Sergio Burdisso
Enhanced for metadata handling and LLM cluster naming
"""
import os
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

# Import LLM utilities from Dialog2Flow
try:
    from util import init_gpt, get_cluster_label
except ModuleNotFoundError:
    # Fallback if running as module
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from util import init_gpt, get_cluster_label

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def load_metadata(metadata_path: str) -> Dict:
    """Load conversations metadata from JSON file."""
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def aggregate_metadata_for_cluster(turn_metadata_list: List[Dict]) -> Dict:
    """
    Aggregate metadata from multiple turns in a cluster.
    
    Numerical fields: Calculate mean (excluding None values)
    Text/List fields: Concatenate unique values
    
    Args:
        turn_metadata_list: List of turn metadata dictionaries
        
    Returns:
        Aggregated metadata dictionary
    """
    if not turn_metadata_list:
        return {}
    
    # Fields to aggregate
    numerical_fields = ['escalation_level', 'churn_risk_score', 'empathy_score']
    list_fields = ['escalation_risks', 'intents_emotions', 'dialogue_acts', 
                   'action_type', 'escalation_reason_tags']
    
    aggregated = {}
    
    # Aggregate numerical fields (mean)
    for field in numerical_fields:
        values = [t.get(field) for t in turn_metadata_list if t.get(field) is not None]
        if values:
            aggregated[field] = float(np.mean(values))
            aggregated[f'{field}_std'] = float(np.std(values))
            aggregated[f'{field}_min'] = float(np.min(values))
            aggregated[f'{field}_max'] = float(np.max(values))
        else:
            aggregated[field] = None
            aggregated[f'{field}_std'] = None
            aggregated[f'{field}_min'] = None
            aggregated[f'{field}_max'] = None
    
    # Aggregate list/text fields (unique concatenation)
    for field in list_fields:
        all_values = []
        for t in turn_metadata_list:
            val = t.get(field)
            if val:
                if isinstance(val, list):
                    all_values.extend(val)
                else:
                    all_values.append(val)
        # Get unique values while preserving order
        unique_values = []
        seen = set()
        for v in all_values:
            if v and v not in seen:
                unique_values.append(v)
                seen.add(v)
        aggregated[field] = unique_values
    
    # Add count of turns in cluster
    aggregated['num_turns_in_cluster'] = len(turn_metadata_list)
    
    return aggregated


def read_dialogues_with_metadata(path: str, metadata: Dict) -> Tuple[List, List, List]:
    """
    Read dialogues from directory and extract metadata.
    
    Args:
        path: Path to directory containing dialogue txt files
        metadata: Dictionary mapping transcript_id to metadata
        
    Returns:
        Tuple of (dialogues, turns, turns_metadata)
    """
    dialogues = []
    all_turns = []
    all_turns_metadata = []
    
    for filename in sorted(os.listdir(path)):
        if not filename.endswith('.txt'):
            continue
        
        # Extract transcript_id from filename (remove .txt extension)
        transcript_id = filename[:-4]
        
        # Read dialogue
        dialogue_path = os.path.join(path, filename)
        with open(dialogue_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Get metadata for this conversation
        conv_metadata = metadata.get(transcript_id, {})
        turns_metadata = conv_metadata.get('turns', [])
        
        # Parse turns
        dialogue_turns = []
        dialogue_metadata = []
        
        for idx, line in enumerate(lines):
            if ':' in line:
                speaker, text = line.split(':', 1)
                dialogue_turns.append({
                    'speaker': speaker.strip(),
                    'text': text.strip(),
                    'dialogue_id': transcript_id,
                    'turn_idx': idx
                })
                
                # Get metadata for this turn
                if idx < len(turns_metadata):
                    dialogue_metadata.append(turns_metadata[idx])
                else:
                    # No metadata available for this turn
                    dialogue_metadata.append({})
        
        if dialogue_turns:
            dialogues.append({
                'dialogue_id': transcript_id,
                'turns': dialogue_turns,
                'metadata': conv_metadata
            })
            all_turns.extend(dialogue_turns)
            all_turns_metadata.extend(dialogue_metadata)
    
    return dialogues, all_turns, all_turns_metadata


def extract_trajectories_with_metadata(
    input_dir: str,
    output_path: str,
    metadata_path: str,
    model_name: str = "sergioburdisso/dialog2flow-joint-bert-base",
    n_clusters: int = None,
    distance_threshold: float = 0.3,
    labels_enabled: bool = False,
    labels_model: str = "llama3:8b",
    labels_top_k: int = 5
):
    """
    Extract trajectories from dialogues with metadata propagation.
    Clusters Agent and Customer utterances SEPARATELY following Dialog2Flow methodology.
    
    Args:
        input_dir: Directory containing dialogue txt files
        output_path: Path to save trajectories JSON
        metadata_path: Path to conversations_metadata.json
        model_name: Name of sentence transformer model
        n_clusters: Number of clusters (if None, uses distance_threshold)
        distance_threshold: Distance threshold for clustering (default 0.3)
        labels_enabled: Whether to generate cluster labels using LLM
        labels_model: LLM model name (ollama or OpenAI)
        labels_top_k: Number of top utterances to use for label generation
    """
    logger.info(f"Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)
    
    logger.info(f"Reading dialogues from {input_dir}")
    dialogues, turns, turns_metadata = read_dialogues_with_metadata(input_dir, metadata)
    
    logger.info(f"Loaded {len(dialogues)} dialogues with {len(turns)} total turns")
    
    # Extract turn texts for embedding
    turn_texts = [f"{t['speaker']}: {t['text']}" for t in turns]
    
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info("Encoding turns...")
    embeddings = model.encode(turn_texts, show_progress_bar=True)
    embeddings = np.array(embeddings)
    
    # Normalize speaker names to 'Agent' and 'Customer'
    logger.info("Normalizing speaker names...")
    speaker_array = np.array([t['speaker'] for t in turns])
    normalized_speakers = np.array(['Agent' if 'Agent' in s else 'Customer' for s in speaker_array])
    
    # Update turns with normalized speaker
    for turn, norm_speaker in zip(turns, normalized_speakers):
        turn['normalized_speaker'] = norm_speaker
    
    # Initialize LLM if labels are enabled
    if labels_enabled:
        logger.info(f"Initializing LLM for cluster labeling: {labels_model}")
        try:
            init_gpt(labels_model)
            logger.info("✓ LLM initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            logger.warning("Continuing without cluster labels...")
            labels_enabled = False
    
    # Cluster Agent and Customer utterances SEPARATELY (Dialog2Flow methodology)
    logger.info("\n=== CLUSTERING AGENT AND CUSTOMER SEPARATELY (Dialog2Flow) ===")
    all_cluster_ids = np.full(len(turns), '', dtype=object)  # Use object dtype for string IDs
    cluster_metadata = {}
    cluster_labels = {}  # Store generated labels
    total_clusters = 0
    
    for speaker in ['Agent', 'Customer']:
        logger.info(f"\nClustering {speaker.upper()} utterances...")
        speaker_mask = normalized_speakers == speaker
        speaker_indices = np.where(speaker_mask)[0]
        
        if not speaker_mask.any():
            logger.warning(f"  WARNING: No {speaker} utterances found!")
            continue
        
        speaker_embeddings = embeddings[speaker_mask]
        speaker_texts = [turns[i]['text'] for i in speaker_indices]
        logger.info(f"  Found {len(speaker_embeddings)} {speaker} utterances")
        
        # Cluster this speaker's utterances
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold if n_clusters is None else None,
            metric='cosine',
            linkage='average'
        )
        
        speaker_labels = clustering.fit_predict(speaker_embeddings)
        n_speaker_clusters = len(set(speaker_labels))
        logger.info(f"  Created {n_speaker_clusters} {speaker} clusters")
        
        # Generate labels for each cluster if enabled
        if labels_enabled:
            logger.info(f"  Generating labels for {n_speaker_clusters} clusters...")
            cluster_label_map = {}
            
            for cluster_id in tqdm(range(n_speaker_clusters), desc=f"Labels ({speaker})"):
                # Get utterances in this cluster
                cluster_mask = speaker_labels == cluster_id
                cluster_utterances = [speaker_texts[i] for i, in_cluster in enumerate(cluster_mask) if in_cluster]
                
                # Get top-k most representative utterances
                top_k_utterances = cluster_utterances[:min(labels_top_k, len(cluster_utterances))]
                
                try:
                    # Generate label using LLM
                    label = get_cluster_label(top_k_utterances, labels_model)
                    cluster_label_map[cluster_id] = label
                    logger.info(f"    Cluster {cluster_id}: \"{label}\"")
                except Exception as e:
                    logger.warning(f"    Failed to generate label for cluster {cluster_id}: {e}")
                    cluster_label_map[cluster_id] = f"cluster_{cluster_id}"
        
        # Assign cluster IDs with speaker prefix (a0, a1... for Agent, c0, c1... for Customer)
        speaker_prefix = speaker[0].lower()  # 'a' for Agent, 'c' for Customer
        
        for local_idx, global_idx in enumerate(speaker_indices):
            local_cluster_id = speaker_labels[local_idx]
            # Create unique cluster ID with speaker prefix
            cluster_id = f"{speaker_prefix}{local_cluster_id}"
            all_cluster_ids[global_idx] = cluster_id
            turns[global_idx]['cluster_id'] = cluster_id
            turns[global_idx]['cluster_speaker'] = speaker
            
            # Store label if generated
            if labels_enabled and local_cluster_id in cluster_label_map:
                cluster_labels[cluster_id] = cluster_label_map[local_cluster_id]
        
        total_clusters += n_speaker_clusters
    
    logger.info(f"\n✓ Total clusters created: {total_clusters}")
    logger.info(f"  (Agent clusters + Customer clusters, kept separate)")
    
    # Aggregate metadata for each cluster (speaker-homogeneous)
    logger.info("\nAggregating metadata for clusters...")
    cluster_turns = defaultdict(list)
    cluster_turn_indices = defaultdict(list)
    
    for idx, (turn, turn_meta) in enumerate(zip(turns, turns_metadata)):
        cluster_id = turn['cluster_id']
        cluster_turns[cluster_id].append(turn_meta)
        cluster_turn_indices[cluster_id].append(idx)
    
    for cluster_id, turn_metas in cluster_turns.items():
        agg_meta = aggregate_metadata_for_cluster(turn_metas)
        # Add speaker information to cluster metadata
        agg_meta['speaker'] = turns[cluster_turn_indices[cluster_id][0]]['cluster_speaker']
        
        # Add generated label if available
        if cluster_id in cluster_labels:
            agg_meta['cluster_label'] = cluster_labels[cluster_id]
        
        cluster_metadata[cluster_id] = agg_meta
    
    logger.info(f"✓ Aggregated metadata for {len(cluster_metadata)} clusters")
    
    # Build trajectories
    logger.info("\nBuilding trajectories...")
    trajectories = []
    
    for dialogue in dialogues:
        trajectory = []
        for turn in dialogue['turns']:
            trajectory.append({
                'cluster_id': turn['cluster_id'],
                'speaker': turn['normalized_speaker'],
                'original_speaker': turn['speaker'],
                'text': turn['text'],
                'turn_idx': turn['turn_idx']
            })
        
        trajectories.append({
            'dialogue_id': dialogue['dialogue_id'],
            'trajectory': trajectory,
            'metadata': dialogue['metadata']
        })
    
    # Prepare output
    output = {
        'model': model_name,
        'n_clusters': total_clusters,
        'distance_threshold': distance_threshold,
        'cluster_metadata': cluster_metadata,
        'cluster_labels': cluster_labels if labels_enabled else {},
        'trajectories': trajectories,
        'clustering_method': 'speaker_separated',
        'labels_enabled': labels_enabled,
        'labels_model': labels_model if labels_enabled else None,
        'note': 'Agent and Customer utterances clustered separately (Dialog2Flow methodology)'
    }
    
    # Save trajectories
    logger.info(f"\nSaving trajectories to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ SUCCESSFULLY EXTRACTED TRAJECTORIES")
    logger.info(f"{'='*60}")
    logger.info(f"  Dialogues processed: {len(trajectories)}")
    logger.info(f"  Total clusters: {total_clusters}")
    logger.info(f"  Clustering method: SPEAKER-SEPARATED (Dialog2Flow)")
    logger.info(f"  Threshold: {distance_threshold}")
    if labels_enabled:
        logger.info(f"  Cluster labels: {len(cluster_labels)} labels generated")
        logger.info(f"  LLM model: {labels_model}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"{'='*60}\n")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Extract trajectories from dialogues with metadata propagation'
    )
    parser.add_argument(
        '--input-dir',
        default='/home/pushpendras0026/dialog2flow/data/example',
        help='Directory containing dialogue txt files'
    )
    parser.add_argument(
        '--output',
        default='/home/pushpendras0026/dialog2flow/output/trajectories_with_metadata.json',
        help='Output path for trajectories JSON'
    )
    parser.add_argument(
        '--metadata',
        default='/home/pushpendras0026/dialog2flow/data/example/conversations_metadata.json',
        help='Path to conversations_metadata.json'
    )
    parser.add_argument(
        '--model',
        default='sergioburdisso/dialog2flow-joint-bert-base',
        help='Sentence transformer model name'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters (if None, uses distance_threshold)'
    )
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=0.3,
        help='Distance threshold for clustering (used if n_clusters is None, default: 0.3)'
    )
    parser.add_argument(
        '-l', '--labels-enabled',
        action='store_true',
        help='Enable LLM-based cluster label generation'
    )
    parser.add_argument(
        '-lm', '--labels-model',
        default='llama3:8b',
        help='LLM model for label generation (default: llama3:8b, supports ollama or OpenAI models)'
    )
    parser.add_argument(
        '--labels-top-k',
        type=int,
        default=5,
        help='Number of top utterances to use for label generation (default: 5)'
    )
    
    args = parser.parse_args()
    
    extract_trajectories_with_metadata(
        args.input_dir,
        args.output,
        args.metadata,
        args.model,
        args.n_clusters,
        args.distance_threshold,
        args.labels_enabled,
        args.labels_model,
        args.labels_top_k
    )


if __name__ == '__main__':
    main()
