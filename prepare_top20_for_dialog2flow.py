#!/usr/bin/env python3
"""
Prepare top 20 conversations from data/top_20 for Dialog2Flow pipeline.
This script:
1. Reads metadata-rich JSON files from data/top_20
2. Extracts speaker and text for Dialog2Flow format
3. Saves to data/example as .txt files
4. Maintains mapping from transcript_id + turn_idx to original metadata

Copyright (c) 2024
MIT License
"""
import os
import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_dialogs_from_top20(top20_dir: str, output_dir: str, metadata_file: str) -> Dict:
    """
    Extract dialogs from top_20 directory and prepare for Dialog2Flow.
    
    Args:
        top20_dir: Directory containing top 20 transcript txt files
        output_dir: Directory to save simplified txt files (data/example)
        metadata_file: Path to save metadata mapping
        
    Returns:
        Dictionary mapping transcript_id to metadata
    """
    logger.info(f"Reading top 20 conversations from {top20_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original JSON to get full metadata
    # Find the summary JSON file
    json_files = [f for f in os.listdir(top20_dir) if f.endswith('.json') and f.startswith('top_20_')]
    
    if not json_files:
        logger.error(f"No summary JSON file found in {top20_dir}")
        return {}
    
    # Use the most recent JSON file
    json_file = sorted(json_files)[-1]
    json_path = os.path.join(top20_dir, json_file)
    
    logger.info(f"Loading summary from {json_path}")
    with open(json_path, 'r') as f:
        summary_data = json.load(f)
    
    conversations = summary_data.get('results', [])
    logger.info(f"Found {len(conversations)} conversations in summary")
    
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
    
    logger.info(f"✓ Prepared {len(conversations_metadata)} conversations for Dialog2Flow")
    logger.info(f"✓ Files saved to {output_dir}")
    logger.info(f"✓ Metadata saved to {metadata_file}")
    
    return conversations_metadata


def main():
    """Main function to prepare top 20 for Dialog2Flow."""
    top20_dir = "/home/pushpendras0026/dialog2flow/data/top_20"
    output_dir = "/home/pushpendras0026/dialog2flow/data/example"
    metadata_file = os.path.join(output_dir, "conversations_metadata.json")
    
    # Clear existing files in output directory (except metadata from previous runs)
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.txt'):
                os.remove(os.path.join(output_dir, filename))
                logger.info(f"Removed old file: {filename}")
    
    # Extract and prepare dialogs
    metadata = extract_dialogs_from_top20(top20_dir, output_dir, metadata_file)
    
    return metadata


if __name__ == '__main__':
    main()
