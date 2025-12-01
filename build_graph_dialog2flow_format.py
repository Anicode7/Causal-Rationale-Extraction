#!/usr/bin/env python3
"""
Build Dialog2Flow-compatible graph from trajectories with metadata.
This converts our trajectory format to the original Dialog2Flow format
and uses the original build_graph.py to create the action flow graph.

Based on Dialog2Flow by Sergio Burdisso
"""
import os
import json
import argparse
import logging
from pathlib import Path
from build_graph import trajectory2graph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def convert_to_dialog2flow_format(trajectories_path: str, output_path: str, domain: str = "banking"):
    """
    Convert our trajectory format to Dialog2Flow format.
    
    Original Dialog2Flow format:
    {
      "dialog_id": {
        "goal": {"domain": ...},
        "log": [
          {"turn": "[start]"},
          {"turn": "SYSTEM: cluster_name"},
          {"turn": "USER: cluster_name"},
          ...
          {"turn": "[end]"}
        ]
      }
    }
    
    Args:
        trajectories_path: Path to our trajectories_with_metadata.json
        output_path: Path to save Dialog2Flow-compatible trajectories
        domain: Domain name for the dialogues
    """
    logger.info(f"Loading trajectories from {trajectories_path}")
    with open(trajectories_path, 'r') as f:
        data = json.load(f)
    
    trajectories = data.get('trajectories', [])
    cluster_metadata = data.get('cluster_metadata', {})
    cluster_labels = data.get('cluster_labels', {})
    
    logger.info(f"Converting {len(trajectories)} trajectories to Dialog2Flow format")
    if cluster_labels:
        logger.info(f"Found {len(cluster_labels)} LLM-generated cluster labels")
    
    # Convert to Dialog2Flow format
    dialog2flow_data = {}
    
    for traj in trajectories:
        dialogue_id = traj['dialogue_id']
        trajectory = traj['trajectory']
        
        # Create log with start/end tokens
        log = [{"turn": "[start]"}]
        
        for turn in trajectory:
            cluster_id = turn['cluster_id']
            speaker = turn['speaker'].upper()  # AGENT or CUSTOMER
            text = turn.get('text', '')
            
            # Get cluster metadata for label
            cluster_meta = cluster_metadata.get(cluster_id, {})
            
            # Use LLM-generated label if available, otherwise use utterance text
            if cluster_id in cluster_labels:
                # Use LLM label
                label = cluster_labels[cluster_id]
            else:
                # Fallback to short utterance text
                label = text[:50].replace(':', '').replace('\n', ' ').strip()
                if len(text) > 50:
                    label += '...'
            
            # Dialog2Flow format: "speaker: cluster_id_label"
            # Extract numeric cluster ID (e.g., "a0" -> "0", "c5" -> "5")
            numeric_id = cluster_id[1:]  # Remove 'a' or 'c' prefix
            
            # Dialog2Flow expects "system:" and "user:" (lowercase)
            speaker_normalized = "system" if speaker == "AGENT" else "user"
            
            # Format: "system: 0_cluster_label" or "user: 5_cluster_label"
            turn_text = f"{speaker_normalized}: {numeric_id}_{label}"
            
            log.append({"turn": turn_text})
        
        log.append({"turn": "[end]"})
        
        # Create Dialog2Flow dialogue structure
        dialog2flow_data[dialogue_id] = {
            "goal": {domain: {}},
            "log": log
        }
    
    # Save in Dialog2Flow format
    logger.info(f"Saving Dialog2Flow-compatible trajectories to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dialog2flow_data, f, indent=2)
    
    logger.info(f"✓ Converted {len(dialog2flow_data)} dialogues to Dialog2Flow format")
    return output_path


def build_dialog2flow_graph(
    trajectories_path: str,
    output_folder: str,
    domain: str = "banking",
    edges_weight: str = "prob-out",
    prune_threshold_edges: float = 0.05,
    prune_threshold_nodes: float = 0.023,
    png_visualization: bool = True,
    interactive_visualization: bool = True
):
    """
    Build Dialog2Flow action flow graph from trajectories.
    
    Args:
        trajectories_path: Path to trajectories_with_metadata.json
        output_folder: Folder to save graph outputs
        domain: Domain name
        edges_weight: Edge weight policy (max, max-out, prob-out)
        prune_threshold_edges: Threshold for pruning edges
        prune_threshold_nodes: Threshold for pruning nodes
        png_visualization: Generate PNG visualization
        interactive_visualization: Generate interactive HTML visualization
    """
    # Convert to Dialog2Flow format
    dialog2flow_trajectories_path = os.path.join(
        os.path.dirname(trajectories_path),
        "trajectories_dialog2flow_format.json"
    )
    
    convert_to_dialog2flow_format(
        trajectories_path,
        dialog2flow_trajectories_path,
        domain
    )
    
    # Use original Dialog2Flow graph builder
    logger.info("Building Dialog2Flow action flow graph...")
    graph, nodes = trajectory2graph(
        path_trajectories=dialog2flow_trajectories_path,
        output_folder=output_folder,
        edges_weight=edges_weight,
        prune_threshold_edges=prune_threshold_edges,
        prune_threshold_nodes=prune_threshold_nodes,
        png_show_ids=True,
        png_visualization=png_visualization,
        interactive_visualization=interactive_visualization
    )
    
    logger.info(f"✓ Graph created: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    logger.info(f"✓ Output saved to: {output_folder}")
    
    return graph, nodes


def main():
    parser = argparse.ArgumentParser(
        description='Build Dialog2Flow action flow graph from trajectories with metadata'
    )
    parser.add_argument(
        '--trajectories',
        default='/home/pushpendras0026/dialog2flow/output/trajectories_with_metadata.json',
        help='Path to trajectories_with_metadata.json'
    )
    parser.add_argument(
        '--output',
        default='/home/pushpendras0026/dialog2flow/output/graph_dialog2flow',
        help='Output folder for graph'
    )
    parser.add_argument(
        '--domain',
        default='banking',
        help='Domain name'
    )
    parser.add_argument(
        '--edges-weight',
        default='prob-out',
        choices=['max', 'max-out', 'prob-out'],
        help='Edge weight policy'
    )
    parser.add_argument(
        '--prune-edges',
        type=float,
        default=0.05,
        help='Threshold for pruning edges'
    )
    parser.add_argument(
        '--prune-nodes',
        type=float,
        default=0.023,
        help='Threshold for pruning nodes'
    )
    parser.add_argument(
        '--no-png',
        action='store_true',
        help='Disable PNG visualization'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive HTML visualization'
    )
    
    args = parser.parse_args()
    
    build_dialog2flow_graph(
        args.trajectories,
        args.output,
        args.domain,
        args.edges_weight,
        args.prune_edges,
        args.prune_nodes,
        not args.no_png,
        not args.no_interactive
    )


if __name__ == '__main__':
    main()
