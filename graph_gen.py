#!/usr/bin/env python3
"""
Integrated Pipeline: Top K Conversations → Dialog2Flow Graph with Metadata

This pipeline orchestrates the complete workflow:
1. Find top K most relevant conversations for a query and domain
2. Store them in data/top_K with full metadata
3. Run the original Dialog2Flow text-only trajectory extraction
4. Attach metadata back onto the Dialog2Flow trajectories
5. Build both metadata-aware and Dialog2Flow graphs (with optional visualizations)
"""
import os
import sys
import json
import shutil
import argparse
import logging
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import query_splitter 
import llm_handler

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query_embeddings_db import CachedEmbeddingSearch
from prepare_for_dialog2flow import extract_dialogs
from extract_trajectories import dialog2trajectories
from extract_trajectories_with_metadata import aggregate_metadata_for_cluster
from build_graph_with_metadata import build_and_export_graph
from build_graph import trajectory2graph, DEFAULT_TOKEN_START, DEFAULT_TOKEN_END

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class GraphGenerator:
    """Integrated pipeline for conversation analysis and graph extraction."""
    def __init__(
        self,
        data_path: str = 'data/final_annotated_dataset.json',
        topK_dir: str = 'data/top_K',
        example_dir: str = 'data/example',
        output_dir: str = 'output',
        embeddings_db: str = None,
        use_cached_embeddings: bool = True
    ):
        self.base_dir = Path(__file__).resolve().parent
        self.data_path = self._resolve_path(data_path)
        self.topK_dir = self._resolve_path(topK_dir)
        self.example_dir = self._resolve_path(example_dir)
        self.output_dir = self._resolve_path(output_dir)
        
        # Determine embeddings database path
        if embeddings_db is None:
            data_dir = os.path.dirname(self.data_path)
            embeddings_db = os.path.join(data_dir, 'embeddings.db')

        self.embeddings_db = self._resolve_path(embeddings_db)
        self.use_cached_embeddings = use_cached_embeddings
        
        # Check if embeddings DB exists
        if use_cached_embeddings:
            if not os.path.exists(self.embeddings_db):
                logger.warning(f"Embeddings database not found at {self.embeddings_db}")
                logger.warning("Will fall back to computing embeddings on-the-fly")
                logger.warning(f"Run 'python3 create_embeddings_db.py --data-path {self.data_path}' to create it")
                self.use_cached_embeddings = False
            else:
                logger.info(f"Using cached embeddings from: {self.embeddings_db}")

        # Keep track of latest run context for downstream steps
        self.last_dialog2flow_path: Optional[str] = None
        self.last_model_name: Optional[str] = None
        self.last_distance_threshold: Optional[float] = None
        self.last_n_clusters: Optional[int] = None
        self.last_domain: Optional[str] = None
        self.last_intent: Optional[str] = None
    
    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        """Return absolute path derived from repo root when input is relative."""
        if path is None:
            return None
        candidate = Path(path)
        if candidate.is_absolute():
            return str(candidate)
        return str((self.base_dir / candidate).resolve())

    def _sanitize_token(self, value: Optional[str], default: str = 'all') -> str:
        """Return lowercase slug for filenames."""
        if not value:
            return default
        token = re.sub(r'[^a-z0-9]+', '_', value.strip().lower()).strip('_')
        return token or default

    def clean_directories(self):
        """Clean previous run outputs."""
        logger.info("=" * 80)
        logger.info("STEP 0: Cleaning previous run outputs")
        logger.info("=" * 80)
        
        # Clear top_K directory
        if os.path.exists(self.topK_dir):
            logger.info(f"Removing files from {self.topK_dir}")
            for filename in os.listdir(self.topK_dir):
                filepath = os.path.join(self.topK_dir, filename)
                try:
                    if os.path.isfile(filepath) or os.path.islink(filepath):
                        os.remove(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception as e:
                    logger.warning(f"Could not remove {filepath}: {e}")
        
        # Clear example directory (keep .gitkeep if exists)
        if os.path.exists(self.example_dir):
            logger.info(f"Removing files from {self.example_dir}")
            for filename in os.listdir(self.example_dir):
                if filename == '.gitkeep':
                    continue
                filepath = os.path.join(self.example_dir, filename)
                try:
                    if os.path.isfile(filepath) or os.path.islink(filepath):
                        os.remove(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception as e:
                    logger.warning(f"Could not remove {filepath}: {e}")
        
        # Clear output directory
        if os.path.exists(self.output_dir):
            logger.info(f"Removing files from {self.output_dir}")
            for filename in os.listdir(self.output_dir):
                if filename.startswith('.'):
                    continue
                filepath = os.path.join(self.output_dir, filename)
                try:
                    if os.path.isfile(filepath) or os.path.islink(filepath):
                        os.remove(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception as e:
                    logger.warning(f"Could not remove {filepath}: {e}")
        
        logger.info("Cleaned directories !!\n")
    
    def step2_prepare_for_dialog2flow(self) -> dict:
        """Step 2: Prepare top K for Dialog2Flow format."""
        logger.info("=" * 80)
        logger.info("STEP 2: Preparing for Dialog2Flow")
        logger.info("=" * 80)
        
        metadata_file = os.path.join(self.example_dir, 'conversations_metadata.json')
        
        # Extract dialogs
        metadata = extract_dialogs(
            self.topK_dir,
            self.example_dir,
            metadata_file
        )
        
        logger.info(f"Prepared {len(metadata)} conversations")
        logger.info(f"Metadata saved to {metadata_file}\n")
        
        return metadata
    
    def step3_extract_dialog2flow_trajectories(
        self,
        model_name: str = "sergioburdisso/dialog2flow-joint-bert-base",
        distance_threshold: float = 0.5,
        domain: Optional[str] = None
    ) -> Dict[str, object]:
        """Step 3: Run the original Dialog2Flow text-only trajectory extraction."""
        logger.info("=" * 80)
        logger.info("STEP 3: Extracting Dialog2Flow Trajectories (Text Only)")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Distance_threshold={distance_threshold}")
        if domain:
            if domain:
                logger.info(f"Input conversations already restricted to domain: {domain}")
            logger.info("")

        os.makedirs(self.output_dir, exist_ok=True)

        # Dialog2Flow uses one threshold per speaker; reuse same value for Agent/Customer
        threshold_value = distance_threshold
        thresholds = [threshold_value, threshold_value]

        logger.info("Invoking dialog2trajectories (text-only)...")
        trajectories_path = dialog2trajectories(
            input_path=self.example_dir,
            output_path=self.output_dir,
            embedding_model=model_name,
            thresholds=thresholds,
            labels_enabled=False,
            dendrogram=False,
            target_domains=None
        )

        if not trajectories_path or not os.path.exists(trajectories_path):
            raise FileNotFoundError("Dialog2Flow trajectories file was not created. Check dialog2trajectories execution.")

        with open(trajectories_path, 'r') as f:
            dialog2flow_output = json.load(f)

        logger.info(f"Extracted trajectories saved to {trajectories_path}\n")
        summary = self._summarize_dialog2flow_output(dialog2flow_output)
        summary['threshold_value'] = threshold_value
        summary['n_clusters_param'] = 'auto'  # since we are using distance thresholding

        self.last_dialog2flow_path = trajectories_path
        self.last_model_name = model_name
        self.last_distance_threshold = distance_threshold
        self.last_n_clusters = None
        self.last_domain = domain

        logger.info(f"Dialogues processed: {summary['dialogue_count']}")
        logger.info(f"Agent clusters: {summary['agent_cluster_count']} | Customer clusters: {summary['customer_cluster_count']}")
        logger.info(f"Trajectories saved to {trajectories_path}\n")

        return {
            'path': trajectories_path,
            'summary': summary,
        }

    def _summarize_dialog2flow_output(self, data: Dict[str, Dict]) -> Dict[str, object]:
        """Collect high-level stats from Dialog2Flow output."""
        cluster_id_sets = {'Agent': set(), 'Customer': set()}

        for dialogue in data.values():
            for turn in dialogue.get('log', []):
                turn_value = turn.get('turn')
                parsed = self._parse_turn_cluster(turn_value)
                if not parsed:
                    continue
                cluster_id_sets[parsed['normalized_speaker']].add(parsed['cluster_id'])

        return {
            'dialogue_count': len(data),
            'agent_cluster_count': len(cluster_id_sets['Agent']),
            'customer_cluster_count': len(cluster_id_sets['Customer']),
            'cluster_ids': {
                'Agent': sorted(cluster_id_sets['Agent']),
                'Customer': sorted(cluster_id_sets['Customer'])
            }
        }

    @staticmethod
    def _normalize_speaker(raw_speaker: Optional[str]) -> str:
        if not raw_speaker:
            return 'Customer'
        speaker = raw_speaker.strip().lower()
        return 'Agent' if speaker in {'system', 'agent'} else 'Customer'

    def _parse_turn_cluster(self, turn_value: Optional[str]) -> Optional[Dict[str, str]]:
        """Parse Dialog2Flow turn string into cluster id, label, and normalized speaker."""
        if not turn_value or turn_value in {DEFAULT_TOKEN_START, DEFAULT_TOKEN_END}:
            return None

        speaker_raw, sep, remainder = turn_value.partition(':')
        if not sep:
            return None

        normalized_speaker = self._normalize_speaker(speaker_raw)
        remainder = remainder.strip()
        if not remainder:
            return None

        if '_' in remainder:
            cluster_numeric, cluster_label = remainder.split('_', 1)
        else:
            cluster_numeric, cluster_label = remainder, remainder

        prefix = 'a' if normalized_speaker == 'Agent' else 'u'
        cluster_id = f"{prefix}{cluster_numeric.strip()}"

        return {
            'cluster_id': cluster_id,
            'cluster_numeric': cluster_numeric.strip(),
            'cluster_label': cluster_label.strip(),
            'normalized_speaker': normalized_speaker
        }

    def step4_attach_metadata_to_trajectories(self) -> Dict[str, object]:
        """Step 4: Re-attach metadata to Dialog2Flow trajectories."""
        logger.info("=" * 80)
        logger.info("STEP 4: Attaching Metadata to Dialog2Flow Trajectories")
        logger.info("=" * 80)

        if not self.last_dialog2flow_path:
            raise RuntimeError("Dialog2Flow trajectories path unavailable. Run step3 first.")

        metadata_path = os.path.join(self.example_dir, 'conversations_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        payload, turn_mapping = self._build_metadata_payload(
            dialog2flow_path=self.last_dialog2flow_path,
            metadata_path=metadata_path
        )

        enriched_path = os.path.join(self.output_dir, 'trajectories_with_metadata.json')
        mapping_path = os.path.join(self.output_dir, 'turn_to_cluster_mapping.json')

        with open(enriched_path, 'w') as f:
            json.dump(payload, f, indent=2)

        with open(mapping_path, 'w') as f:
            json.dump(turn_mapping, f, indent=2)

        logger.info(f"Metadata attached to {payload['dialogue_count']} dialogues")
        logger.info(f"Clusters with metadata: {payload['n_clusters']} (Agent={payload['n_agent_clusters']}, Customer={payload['n_customer_clusters']})")
        logger.info(f"Saved enriched trajectories to {enriched_path}")
        logger.info(f"Turn → cluster mapping saved to {mapping_path}\n")

        return {
            'path': enriched_path,
            'mapping_path': mapping_path,
            'summary': {
                'dialogues': payload['dialogue_count'],
                'clusters': payload['n_clusters']
            }
        }

    def _build_metadata_payload(
        self,
        dialog2flow_path: str,
        metadata_path: str
    ) -> Tuple[Dict[str, object], Dict[str, List[Dict[str, object]]]]:
        """Combine Dialog2Flow outputs with stored metadata."""
        with open(dialog2flow_path, 'r') as f:
            dialog2flow_data = json.load(f)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        cluster_turns: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        cluster_labels: Dict[str, str] = {}
        trajectories: List[Dict[str, object]] = []
        turn_mapping: Dict[str, List[Dict[str, object]]] = {}
        skipped_dialogues: List[str] = []

        for dialogue_id, dialogue in dialog2flow_data.items():
            conversation_meta = metadata.get(dialogue_id)
            if not conversation_meta:
                logger.warning(f"Metadata missing for dialogue {dialogue_id}; skipping metadata enrichment for this conversation")
                skipped_dialogues.append(dialogue_id)
                continue

            turns_meta = conversation_meta.get('turns', [])
            dialog_turns = dialogue.get('log', [])[1:-1]  # Skip [start]/[end]

            if len(turns_meta) != len(dialog_turns):
                logger.warning(
                    f"Turn count mismatch for {dialogue_id}: Dialog2Flow={len(dialog_turns)} vs metadata={len(turns_meta)}. Using minimum available."
                )

            aligned_length = min(len(turns_meta), len(dialog_turns))
            mapping_entries: List[Dict[str, object]] = []
            trajectory_entries: List[Dict[str, object]] = []

            for idx in range(aligned_length):
                turn_value = dialog_turns[idx].get('turn')
                parsed = self._parse_turn_cluster(turn_value)
                if not parsed:
                    continue

                cluster_id = parsed['cluster_id']
                cluster_labels.setdefault(cluster_id, parsed['cluster_label'])

                turn_metadata = turns_meta[idx].copy() if idx < len(turns_meta) else {}
                turn_metadata.setdefault('turn_idx', idx)
                turn_metadata.setdefault('speaker', parsed['normalized_speaker'])
                turn_metadata.setdefault('text', turn_metadata.get('text', ''))
                turn_metadata['dialogue_id'] = dialogue_id

                cluster_turns[cluster_id].append(turn_metadata)

                mapping_entries.append({
                    'turn_idx': turn_metadata.get('turn_idx', idx),
                    'cluster_id': cluster_id,
                    'speaker': turn_metadata.get('speaker'),
                    'normalized_speaker': parsed['normalized_speaker'],
                    'text': turn_metadata.get('text', '')
                })

                trajectory_entries.append({
                    'cluster_id': cluster_id,
                    'speaker': parsed['normalized_speaker'],
                    'text': turn_metadata.get('text', ''),
                    'turn_idx': turn_metadata.get('turn_idx', idx),
                    'metadata': turn_metadata
                })

            if mapping_entries:
                conversation_level_meta = {k: v for k, v in conversation_meta.items() if k != 'turns'}
                trajectories.append({
                    'dialogue_id': dialogue_id,
                    'trajectory': trajectory_entries,
                    'metadata': conversation_level_meta
                })
                turn_mapping[dialogue_id] = mapping_entries

        cluster_metadata: Dict[str, Dict[str, object]] = {}
        for cluster_id, turns in cluster_turns.items():
            aggregated = aggregate_metadata_for_cluster(turns)
            aggregated['cluster_label'] = cluster_labels.get(cluster_id)
            aggregated['speaker'] = 'Agent' if cluster_id.startswith('a') else 'Customer'
            aggregated['utterance_sources'] = [
                {
                    'transcript_id': t.get('dialogue_id'),
                    'turn_idx': t.get('turn_idx'),
                    'speaker': t.get('speaker'),
                    'text': t.get('text', '')
                }
                for t in turns
            ]
            cluster_metadata[cluster_id] = aggregated

        payload = {
            'model': self.last_model_name,
            'distance_threshold': self.last_distance_threshold,
            'n_clusters': len(cluster_metadata),
            'n_agent_clusters': sum(1 for cid in cluster_metadata if cid.startswith('a')),
            'n_customer_clusters': sum(1 for cid in cluster_metadata if cid.startswith('u')),
            'cluster_labels': cluster_labels,
            'cluster_metadata': cluster_metadata,
            'trajectories': trajectories,
            'dialogue_count': len(trajectories),
            'params': {
                'n_clusters': self.last_n_clusters,
                'distance_threshold': self.last_distance_threshold,
                'domain': self.last_domain,
                'intent': self.last_intent
            },
            'dialog2flow_trajectory_path': dialog2flow_path,
            'skipped_dialogues': skipped_dialogues,
            'notes': 'Metadata appended after executing the original Dialog2Flow text-only pipeline.'
        }

        return payload, turn_mapping

    def step5_build_graphs(
        self,
        formats: List[str] = ['json', 'graphml', 'html'],
        domain: Optional[str] = None
    ) -> Dict[str, object]:
        """Step 5: Build metadata-aware and Dialog2Flow graphs."""
        logger.info("=" * 80)
        logger.info("STEP 5: Building Graphs")
        logger.info("=" * 80)
        logger.info(f"Export formats: {', '.join(formats)}")
        logger.info("")

        if not self.last_dialog2flow_path or not os.path.exists(self.last_dialog2flow_path):
            raise RuntimeError("Dialog2Flow trajectories missing. Run step3 before building graphs.")

        enriched_path = os.path.join(self.output_dir, 'trajectories_with_metadata.json')
        if not os.path.exists(enriched_path):
            raise FileNotFoundError("Metadata-enriched trajectories not found. Run step4 before building graphs.")

        logger.info("Building Dialog2Flow action-flow graph...")
        d2f_graph, d2f_nodes = trajectory2graph(
            path_trajectories=self.last_dialog2flow_path,
            output_folder=os.path.join(self.output_dir, 'graph_dialog2flow'),
            edges_weight='prob-out',
            prune_threshold_edges=0.1,
            prune_threshold_nodes=0.023,
            png_visualization=True,
            interactive_visualization=True,
            target_domains=None
        )

        logger.info(f"Dialog2Flow graph: {len(d2f_graph.nodes)} nodes, {len(d2f_graph.edges)} edges")

        logger.info("")
        logger.info("Building metadata-enhanced graph...")
        metadata_graph = build_and_export_graph(
            trajectories_path=enriched_path,
            output_dir=self.output_dir,
            formats=formats
        )

        logger.info(f"Metadata graph: {metadata_graph.number_of_nodes()} nodes, {metadata_graph.number_of_edges()} edges")
        logger.info(f"Graph artifacts written to {self.output_dir}\n")

        return {
            'dialog2flow_graph': {
                'graph': d2f_graph,
                'nodes': d2f_nodes,
                'path': self.last_dialog2flow_path
            },
            'metadata_graph': {
                'graph': metadata_graph,
                'path': enriched_path
            }
        }
    
    def run(
        self,
        query: str,
        model_name: str = "sergioburdisso/dialog2flow-joint-bert-base",
        distance_threshold: float = 0.5,
        export_formats: list = ['json', 'graphml'],
        labels_enabled: bool = False,
        labels_model: str = 'llama3:8b',
        clean: bool = True,
        follow_up: int = 0,
        topK: int = 20
    ):
        """
        Run the complete integrated pipeline.
        
        Args:
            query: Search query for finding relevant conversations
            domain: Filter by domain (optional)
            intent: Filter by intent (optional)
            model_name: Sentence transformer model for trajectory extraction
            distance_threshold: Distance threshold for clustering
            export_formats: List of graph export formats
            labels_enabled: Enable LLM-based cluster label generation
            labels_model: LLM model for label generation (default: llama3:8b)
            clean: Whether to clean directories before starting
            follow_up: Whether follow-up queries to consider (default: 0)
            topK: Number of top conversations to retrieve (default: 20)
        
        Returns:
            Dictionary with results from each step
        """
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 15 + "INTEGRATED PIPELINE: TOP K → DIALOG2FLOW" + " " * 22 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info("")
        domain = None
        intent = None
        results = {}
        self.last_intent = intent
        if follow_up > 0:
            clean = False
        try:
            # Step 0: Clean directories
            if clean:
                self.clean_directories()
            
            # Step 1: Find top 20 conversations
            llm = llm_handler.llm()
            results['top_K'], queries_list = query_splitter.categorize_query(query,llm,self.embeddings_db, self.data_path,follow_up=follow_up, topK_dir=self.topK_dir,topk=topK)
            
            # Step 2: Prepare for Dialog2Flow
            results['metadata'] = self.step2_prepare_for_dialog2flow()
            
            # Step 3: Extract trajectories with metadata
            results['dialog2flow'] = self.step3_extract_dialog2flow_trajectories(
                model_name=model_name,
                distance_threshold=distance_threshold,
                domain=domain
            )

            # Step 4: Attach metadata to Dialog2Flow trajectories
            results['trajectories_with_metadata'] = self.step4_attach_metadata_to_trajectories()

            # Step 5: Build graphs
            results['graphs'] = self.step5_build_graphs(export_formats, domain)
            
            return results, queries_list
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


# check what n_clusters is and add it here
def generate_json_graph(query,data_path,distance_threshold = 0.6,follow_up = 0,topK=20):

    pipeline = GraphGenerator(data_path=data_path)
    
    results, queries_list = pipeline.run(
        query=query,
        distance_threshold=distance_threshold,
        follow_up=follow_up,
        topK=topK
    )

    return pipeline, queries_list
