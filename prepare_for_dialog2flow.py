#!/usr/bin/env python3
"""
Prepare top K conversations from data/top_K for Dialog2Flow pipeline.
This script:
1. Reads metadata-rich JSON files from data/top_K
2. Extracts speaker and text for Dialog2Flow format
3. Saves to data/example as .txt files
4. Maintains mapping from transcript_id + turn_idx to original metadata
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_dialogs(topK_dir: str, output_dir: str, metadata_file: str) -> Dict:
    """
    Extract dialogs from top_K directory and prepare for Dialog2Flow.
    
    Args:
        topK_dir: Directory containing top K transcript txt files
        output_dir: Directory to save simplified txt files (data/example)
        metadata_file: Path to save metadata mapping
        
    Returns:
        Dictionary mapping transcript_id to metadata
    """
    logger.info(f"Reading top K conversations from {topK_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original JSON to get full metadata
    # Find the summary JSON file
    # gather all matching json files (sorted for deterministic order)
    json_files = sorted([f for f in os.listdir(topK_dir) if f.endswith('.json') and f.startswith('top_K_')])
    if not json_files:
        logger.error(f"No top_K JSON files found in {topK_dir}")
        return {}

    # Instead of using only the most recent file, process all json files and merge conversations
    conversations = []
    for json_file in json_files:
        json_path = os.path.join(topK_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping {json_file}: failed to read/parse ({e})")
            continue

        # Accept several possible shapes; prefer a top-level 'conversations' list
        if isinstance(data, dict):
            convs = data.get('results',[])
            
        else:
            convs = None

        if convs is None:
            # If the file itself is a list, assume it's the conversations
            if isinstance(data, list):
                convs = data
            else:
                logger.warning(f"Skipping {json_file}: no conversations list found")
                continue

        if not isinstance(convs, list):
            logger.warning(f"Skipping {json_file}: conversations is not a list")
            continue
        
        logger.info(f"Found {len(conversations)} conversations in summary")
        conversations.extend(convs)
    
    # Store all metadata
    conversations_metadata = {}
    
    # Process each conversation
    for conv_idx, conversation in enumerate(conversations):
        transcript_id = conversation['transcript_id']
        logger.info(f"Processing {conv_idx + 1}/{len(conversations)}: {transcript_id}")
        
        # Create simplified txt file for Dialog2Flow
        txt_filename = f"{transcript_id}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        # Store metadata for each turn
        turn_metadata = []
        
        with open(txt_path, 'w') as f:
            for turn_idx, turn in enumerate(conversation['conversation']):
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                
                # Write in Dialog2Flow format: Speaker: text
                f.write(f"{speaker}: {text}\n")
                
                # Store full metadata for this turn
                turn_metadata.append({
                    'turn_idx': turn_idx,
                    'speaker': speaker,
                    'text': text,
                    'escalation_level': turn.get('escalation_level'),
                    'escalation_risks': turn.get('escalation_risks'),
                    'churn_risk_score': turn.get('churn_risk_score'),
                    'empathy_score': turn.get('empathy_score'),
                    'intents_emotions': turn.get('intents_emotions'),
                    'dialogue_acts': turn.get('dialogue_acts'),
                    'action_type': turn.get('action_type'),
                    'escalation_reason_tags': turn.get('escalation_reason_tags', [])
                })
        
        # Store conversation-level metadata
        conversations_metadata[transcript_id] = {
            'transcript_id': transcript_id,
            'domain': conversation.get('domain'),
            'intent': conversation.get('intent'),
            'time_of_interaction': conversation.get('time_of_interaction'),
            'reason_for_call': conversation.get('reason_for_call'),
            'satisfaction_score': conversation.get('satisfaction_score'),
            'satisfaction_label': conversation.get('satisfaction_label'),
            'similarity_metadata': conversation.get('similarity_metadata'),
            'turns': turn_metadata,
            'num_turns': len(turn_metadata)
        }
    
    # Save metadata
    logger.info(f"Saving metadata to {metadata_file}")
    with open(metadata_file, 'w') as f:
        json.dump(conversations_metadata, f, indent=2)
    
    logger.info(f"Prepared {len(conversations_metadata)} conversations for Dialog2Flow")
    logger.info(f"Files saved to {output_dir}")
    logger.info(f"Metadata saved to {metadata_file}")
    
    return conversations_metadata


def main():
    """Main function to prepare top K for Dialog2Flow."""
    base_dir = Path(__file__).resolve().parent
    topK_dir = (base_dir / 'data' / 'top_K').resolve()
    output_dir = (base_dir / 'data' / 'example').resolve()
    metadata_file = output_dir / "conversations_metadata.json"
    
    # Clear existing files in output directory (except metadata from previous runs)
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.txt'):
                os.remove(os.path.join(output_dir, filename))
                logger.info(f"Removed old file: {filename}")
    
    # Extract and prepare dialogs
    metadata = extract_dialogs(str(topK_dir), str(output_dir), str(metadata_file))
    
    return metadata


if __name__ == '__main__':
    main()
