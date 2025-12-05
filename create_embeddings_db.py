#!/usr/bin/env python3
"""
Create embeddings for ALL domains and store them permanently in SQLite database.
Run this script ONCE to create the embedding database.

Usage:
    python3 create_embeddings_db.py --data-path <path_to_json> --db-path <path_to_db>
"""           
# //data/embeddings.db(path to db)

import json
import os
import sqlite3
import numpy as np
import argparse
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm


class EmbeddingDatabaseCreator:
    """Create and manage persistent embedding database."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """Initialize with sentence transformer model."""
        print(f"[INFO] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    def create_database(self, db_path: str):
        """Create SQLite database schema."""
        print(f"[INFO] Creating database at: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Table for turn embeddings and metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                intent TEXT,
                reason_for_call TEXT,
                turn_idx INTEGER NOT NULL,
                speaker TEXT,
                text TEXT NOT NULL,
                combined_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                escalation_level INTEGER,
                churn_risk_score REAL,
                empathy_score REAL,
                intents_emotions TEXT,
                dialogue_acts TEXT,
                action_type TEXT,
                escalation_reason_tags TEXT,
                time_of_interaction TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Index for fast domain filtering
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON embeddings(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transcript_id ON embeddings(transcript_id)')
        
        # Table for metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        print("[INFO] ✓ Database schema created")
        
    def prepare_turn_text(self, turn: Dict, reason_for_call: str = "") -> str:
        """Combine turn text with metadata for embedding."""
        text = turn.get('text', '')
        intents_emotions = turn.get('intents_emotions', '')
        dialogue_acts = turn.get('dialogue_acts', '')
        action_type = turn.get('action_type', '')
        escalation_tags = ' '.join(turn.get('escalation_reason_tags', []))
        
        combined = f"{text} | Intent/Emotion: {intents_emotions} | Dialogue Act: {dialogue_acts} | Action: {action_type} | Tags: {escalation_tags}"
        
        if reason_for_call:
            combined = f"Reason: {reason_for_call} | {combined}"
        
        return combined
    
    def create_embeddings_for_all_data(self, data_path: str, db_path: str):
        """
        Process all data and create embeddings, storing in database.
        
        Args:
            data_path: Path to final_json_for_d2f.json
            db_path: Path to SQLite database
        """
        # Create database
        self.create_database(db_path)
        
        # Load data
        print(f"[INFO] Loading data from: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        print(f"[INFO] Loaded {len(data)} transcripts")
        
        # Prepare all texts and metadata
        all_texts = []
        all_metadata = []
        
        for transcript in tqdm(data, desc="Preparing turns", unit="transcript"):
            transcript_id = transcript.get('transcript_id', '')
            domain = transcript.get('domain', '')
            intent = transcript.get('intent', '')
            reason_for_call = transcript.get('reason_for_call', '')
            time_of_interaction = transcript.get('time_of_interaction', '')
            
            conversation = transcript.get('conversation', [])
            
            for turn_idx, turn in enumerate(conversation):
                combined_text = self.prepare_turn_text(turn, reason_for_call)
                all_texts.append(combined_text)
                
                all_metadata.append({
                    'transcript_id': transcript_id,
                    'domain': domain,
                    'intent': intent,
                    'reason_for_call': reason_for_call,
                    'turn_idx': turn_idx,
                    'speaker': turn.get('speaker', ''),
                    'text': turn.get('text', ''),
                    'combined_text': combined_text,
                    'escalation_level': turn.get('escalation_level', 0),
                    'churn_risk_score': turn.get('churn_risk_score', 0.0),
                    'empathy_score': turn.get('empathy_score', 0.0),
                    'intents_emotions': turn.get('intents_emotions', ''),
                    'dialogue_acts': turn.get('dialogue_acts', ''),
                    'action_type': turn.get('action_type', ''),
                    'escalation_reason_tags': json.dumps(turn.get('escalation_reason_tags', [])),
                    'time_of_interaction': time_of_interaction
                })
        
        print(f"Total turns: {len(all_texts)}")
        
        # Create embeddings in batches
        print("Creating embeddings...")
        embeddings = self.model.encode(all_texts, show_progress_bar=True, batch_size=32)
        print(f"✓ Embeddings shape: {embeddings.shape}")
        
        # Store in database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        created_at = datetime.now().isoformat()
        
        for metadata, embedding in tqdm(zip(all_metadata, embeddings), total=len(all_metadata), desc="Storing in DB", unit="emb"):
            embedding_blob = pickle.dumps(embedding)
            
            cursor.execute('''
                INSERT INTO embeddings (
                    transcript_id, domain, intent, reason_for_call, turn_idx,
                    speaker, text, combined_text, embedding,
                    escalation_level, churn_risk_score, empathy_score,
                    intents_emotions, dialogue_acts, action_type,
                    escalation_reason_tags, time_of_interaction, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata['transcript_id'],
                metadata['domain'],
                metadata['intent'],
                metadata['reason_for_call'],
                metadata['turn_idx'],
                metadata['speaker'],
                metadata['text'],
                metadata['combined_text'],
                embedding_blob,
                metadata['escalation_level'],
                metadata['churn_risk_score'],
                metadata['empathy_score'],
                metadata['intents_emotions'],
                metadata['dialogue_acts'],
                metadata['action_type'],
                metadata['escalation_reason_tags'],
                metadata['time_of_interaction'],
                created_at
            ))
        
        # Store metadata
        cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('model_name', self.model_name))
        cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('total_embeddings', str(len(embeddings))))
        cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('embedding_dim', str(embeddings.shape[1])))
        cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('created_at', created_at))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Stored {len(embeddings):,} embeddings")
        
        # Create and save FAISS index
        faiss_index_path = db_path.replace('.db', '.faiss')
        print(f"Creating FAISS index: {faiss_index_path}")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings.astype('float32'))
        
        faiss.write_index(index, faiss_index_path)
        print(f"[INFO] ✓ FAISS index saved with {index.ntotal} vectors")
        
        # Print statistics
        self.print_statistics(db_path)
        
    def print_statistics(self, db_path: str):
        """Print database statistics."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("\n" + "="*80)
        print("DATABASE STATISTICS")
        print("="*80)
        
        # Total embeddings
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        total = cursor.fetchone()[0]
        print(f"Total embeddings: {total}")
        
        # By domain
        cursor.execute('SELECT domain, COUNT(*) FROM embeddings GROUP BY domain ORDER BY COUNT(*) DESC')
        print("\nEmbeddings by domain:")
        for domain, count in cursor.fetchall():
            print(f"  {domain}: {count}")
        
        # Metadata
        cursor.execute('SELECT key, value FROM metadata')
        print("\nMetadata:")
        for key, value in cursor.fetchall():
            print(f"  {key}: {value}")
        
        print("="*80 + "\n")
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Create embedding database for all domains')
    parser.add_argument('--data-path', type=str, required=False,default = "data/final_annotated_dataset.json",
                        help='Path to final_json_for_d2f.json')
    parser.add_argument('--db-path', type=str, default='data/embeddings.db',
                        help='Path to SQLite database (default: data/embeddings.db)')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='Sentence transformer model name')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    data_path= Path(args.data_path)
    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = (base_dir / db_path).resolve()

    # Create output directory if needed
    os.makedirs(db_path.parent, exist_ok=True)
    
    print("\n" + "="*80)
    print("EMBEDDING DATABASE CREATION")
    print("="*80)
    print(f"Data path: {data_path}")
    print(f"Database path: {db_path}")
    print(f"Model: {args.model}")
    print("="*80 + "\n")
    
    creator = EmbeddingDatabaseCreator(model_name=args.model)
    creator.create_embeddings_for_all_data(str(data_path), str(db_path))
    
    print("\nEMBEDDING DATABASE CREATED SUCCESSFULLY!")
    print(f"   Database: {db_path}")
    print(f"   FAISS index: {str(db_path).replace('.db', '.faiss')}")
    print("\nYou can now run queries using the integrated pipeline.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
