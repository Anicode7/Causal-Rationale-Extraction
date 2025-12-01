#!/usr/bin/env python3
"""
Integrated Pipeline: Top 20 Conversations → Dialog2Flow Graph with Metadata

This pipeline orchestrates the complete workflow:
1. Find top 20 most relevant conversations for a query and domain
2. Store them in data/top_20 with full metadata
3. Extract simplified text format for Dialog2Flow
4. Run trajectory extraction with metadata propagation
5. Build graph with metadata attached to nodes
6. Generate interactive visualization

Copyright (c) 2024
MIT License
"""
import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from find_top_20_conversations import ConversationSearchPipeline
from prepare_top20_for_dialog2flow import extract_dialogs_from_top20
from extract_trajectories_with_metadata import extract_trajectories_with_metadata
from build_graph_with_metadata import build_and_export_graph

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedPipeline:
    """Integrated pipeline for conversation analysis and graph extraction."""
    
    def __init__(
        self,
        data_path: str = '/home/pushpendras0026/dialog2flow/data/final_json_for_d2f.json',
        top20_dir: str = '/home/pushpendras0026/dialog2flow/data/top_20',
        example_dir: str = '/home/pushpendras0026/dialog2flow/data/example',
        output_dir: str = '/home/pushpendras0026/dialog2flow/output'
    ):
        self.data_path = data_path
        self.top20_dir = top20_dir
        self.example_dir = example_dir
        self.output_dir = output_dir
    
    def clean_directories(self):
        """Clean previous run outputs."""
        logger.info("=" * 80)
        logger.info("STEP 0: Cleaning previous run outputs")
        logger.info("=" * 80)
        
        # Clear top_20 directory
        if os.path.exists(self.top20_dir):
            logger.info(f"Removing files from {self.top20_dir}")
            for filename in os.listdir(self.top20_dir):
                filepath = os.path.join(self.top20_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
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
                    if os.path.isfile(filepath):
                        os.remove(filepath)
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
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                except Exception as e:
                    logger.warning(f"Could not remove {filepath}: {e}")
        
        logger.info("✓ Cleaned directories\n")
    
    def step1_find_top20(self, query: str, domain: str = None) -> dict:
        """Step 1: Find top 20 most relevant conversations."""
        logger.info("=" * 80)
        logger.info("STEP 1: Finding Top 20 Conversations")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")
        if domain:
            logger.info(f"Domain: {domain}")
        logger.info("")
        
        # Initialize pipeline
        pipeline = ConversationSearchPipeline()
        
        # Run pipeline
        results = pipeline.run_pipeline(
            json_file_path=self.data_path,
            domain=domain,
            query=query,
            output_dir=self.top20_dir,
            top_k=20
        )
        
        if results:
            logger.info(f"✓ Found {len(results)} relevant conversations")
            logger.info(f"✓ Saved to {self.top20_dir}\n")
        else:
            logger.warning("⚠ No results found")
        
        return results if results else []
    
    def step2_prepare_for_dialog2flow(self) -> dict:
        """Step 2: Prepare top 20 for Dialog2Flow format."""
        logger.info("=" * 80)
        logger.info("STEP 2: Preparing for Dialog2Flow")
        logger.info("=" * 80)
        
        metadata_file = os.path.join(self.example_dir, 'conversations_metadata.json')
        
        # Extract dialogs
        metadata = extract_dialogs_from_top20(
            self.top20_dir,
            self.example_dir,
            metadata_file
        )
        
        logger.info(f"✓ Prepared {len(metadata)} conversations")
        logger.info(f"✓ Metadata saved to {metadata_file}\n")
        
        return metadata
    
    def step3_extract_trajectories(
        self,
        model_name: str = "sergioburdisso/dialog2flow-joint-bert-base",
        n_clusters: int = None,
        distance_threshold: float = 0.5,
        labels_enabled: bool = False,
        labels_model: str = 'llama3:8b'
    ) -> dict:
        """Step 3: Extract trajectories with metadata propagation."""
        logger.info("=" * 80)
        logger.info("STEP 3: Extracting Trajectories with Metadata")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Clustering: n_clusters={n_clusters}, distance_threshold={distance_threshold}")
        if labels_enabled:
            logger.info(f"LLM Labels: ENABLED (model: {labels_model})")
        logger.info("")
        
        trajectories_path = os.path.join(self.output_dir, 'trajectories_with_metadata.json')
        metadata_path = os.path.join(self.example_dir, 'conversations_metadata.json')
        
        # Extract trajectories
        trajectories = extract_trajectories_with_metadata(
            self.example_dir,
            trajectories_path,
            metadata_path,
            model_name,
            n_clusters,
            distance_threshold,
            labels_enabled,
            labels_model
        )
        
        logger.info(f"✓ Extracted {len(trajectories['trajectories'])} trajectories")
        logger.info(f"✓ Created {trajectories['n_clusters']} clusters with metadata")
        if labels_enabled:
            logger.info(f"✓ Generated {len(trajectories.get('cluster_labels', {}))} cluster labels")
        logger.info(f"✓ Saved to {trajectories_path}\n")
        
        return trajectories
    
    def step4_build_graph(
        self,
        formats: list = ['json', 'graphml', 'html'],
        domain: str = 'banking'
    ) -> object:
        """Step 4: Build graph with metadata."""
        logger.info("=" * 80)
        logger.info("STEP 4: Building Graphs")
        logger.info("=" * 80)
        logger.info(f"Export formats: {', '.join(formats)}")
        logger.info("")
        
        trajectories_path = os.path.join(self.output_dir, 'trajectories_with_metadata.json')
        
        # Build metadata-enhanced graph
        logger.info("Building metadata-enhanced graph...")
        graph = build_and_export_graph(
            trajectories_path,
            self.output_dir,
            formats
        )
        
        logger.info(f"✓ Metadata graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Build Dialog2Flow-style action flow graph
        logger.info("\nBuilding Dialog2Flow action flow graph...")
        from build_graph_dialog2flow_format import build_dialog2flow_graph
        
        d2f_graph, _ = build_dialog2flow_graph(
            trajectories_path=trajectories_path,
            output_folder=os.path.join(self.output_dir, 'graph_dialog2flow'),
            domain=domain,
            edges_weight='prob-out',
            prune_threshold_edges=0.05,
            prune_threshold_nodes=0.023,
            png_visualization=True,
            interactive_visualization=True
        )
        
        logger.info(f"✓ Dialog2Flow graph: {len(d2f_graph.nodes)} nodes, {len(d2f_graph.edges)} edges")
        logger.info(f"✓ Exported to {self.output_dir}\n")
        
        return {'metadata_graph': graph, 'dialog2flow_graph': d2f_graph}
    
    def run(
        self,
        query: str,
        domain: str = None,
        model_name: str = "sergioburdisso/dialog2flow-joint-bert-base",
        n_clusters: int = None,
        distance_threshold: float = 0.5,
        export_formats: list = ['json', 'graphml', 'html'],
        labels_enabled: bool = False,
        labels_model: str = 'llama3:8b',
        clean: bool = True
    ):
        """
        Run the complete integrated pipeline.
        
        Args:
            query: Search query for finding relevant conversations
            domain: Filter by domain (optional)
            model_name: Sentence transformer model for trajectory extraction
            n_clusters: Number of clusters (if None, uses distance_threshold)
            distance_threshold: Distance threshold for clustering
            export_formats: List of graph export formats
            labels_enabled: Enable LLM-based cluster label generation
            labels_model: LLM model for label generation (default: llama3:8b)
            clean: Whether to clean directories before starting
        
        Returns:
            Dictionary with results from each step
        """
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 15 + "INTEGRATED PIPELINE: TOP 20 → DIALOG2FLOW" + " " * 22 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info("")
        
        results = {}
        
        try:
            # Step 0: Clean directories
            if clean:
                self.clean_directories()
            
            # Step 1: Find top 20 conversations
            results['top_20'] = self.step1_find_top20(query, domain)
            
            # Step 2: Prepare for Dialog2Flow
            results['metadata'] = self.step2_prepare_for_dialog2flow()
            
            # Step 3: Extract trajectories with metadata
            results['trajectories'] = self.step3_extract_trajectories(
                model_name,
                n_clusters,
                distance_threshold,
                labels_enabled,
                labels_model
            )
            
            # Step 4: Build graph with metadata
            results['graph'] = self.step4_build_graph(export_formats, domain)
            
            # Final summary
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"✓ Top 20 conversations saved to: {self.top20_dir}")
            logger.info(f"✓ Dialog2Flow input saved to: {self.example_dir}")
            logger.info(f"✓ Trajectories and graph saved to: {self.output_dir}")
            logger.info("")
            logger.info("OUTPUT FILES:")
            logger.info(f"  - Top 20 transcripts: {self.top20_dir}/*.txt")
            logger.info(f"  - Metadata: {self.example_dir}/conversations_metadata.json")
            logger.info(f"  - Trajectories: {self.output_dir}/trajectories_with_metadata.json")
            logger.info(f"  - Graph JSON: {self.output_dir}/graph_with_metadata.json")
            if 'graphml' in export_formats:
                logger.info(f"  - Graph GraphML: {self.output_dir}/graph_with_metadata.graphml")
            if 'html' in export_formats:
                logger.info(f"  - Visualization: {self.output_dir}/graph_visualization.html")
            logger.info("")
            logger.info("🎉 All metadata has been preserved and aggregated at cluster/node level!")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Integrated pipeline: Top 20 conversations → Dialog2Flow graph with metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find top 20 flight-related conversations and build graph
  python integrated_pipeline.py --query "flight delay compensation" --domain Flight
  
  # Search across all domains
  python integrated_pipeline.py --query "customer escalation handling"
  
  # Custom clustering parameters
  python integrated_pipeline.py --query "refund request" --n-clusters 15
        """
    )
    
    parser.add_argument(
        '--query',
        required=True,
        help='Search query for finding relevant conversations'
    )
    parser.add_argument(
        '--domain',
        default=None,
        help='Filter by domain (Flight, Hotel, Restaurant, etc.)'
    )
    parser.add_argument(
        '--model',
        default='sergioburdisso/dialog2flow-joint-bert-base',
        help='Sentence transformer model for trajectory extraction'
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
        default=0.5,
        help='Distance threshold for clustering'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['json', 'graphml', 'html'],
        choices=['json', 'graphml', 'gexf', 'html'],
        help='Graph export formats'
    )
    parser.add_argument(
        '-l', '--labels',
        action='store_true',
        help='Enable LLM-based cluster label generation'
    )
    parser.add_argument(
        '-lm', '--labels-model',
        default='llama3:8b',
        help='LLM model for label generation (default: llama3:8b, supports ollama or OpenAI models)'
    )
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Do not clean directories before starting'
    )
    parser.add_argument(
        '--data-path',
        default='/home/pushpendras0026/dialog2flow/data/final_json_for_d2f.json',
        help='Path to input JSON data'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = IntegratedPipeline(data_path=args.data_path)
    
    pipeline.run(
        query=args.query,
        domain=args.domain,
        model_name=args.model,
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
        export_formats=args.formats,
        labels_enabled=args.labels,
        labels_model=args.labels_model,
        clean=not args.no_clean
    )


if __name__ == '__main__':
    main()
