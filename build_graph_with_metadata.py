#!/usr/bin/env python3
"""
Build graph from trajectories with metadata.
This is an enhanced version of Dialog2Flow's build_graph.py that:
1. Reads trajectories with cluster metadata
2. Builds graph with metadata attached to nodes
3. Exports graph with metadata in various formats

Based on Dialog2Flow by Sergio Burdisso
Enhanced for metadata handling
"""
import os
import json
import argparse
from pathlib import Path
import networkx as nx
from typing import Dict, List
from collections import defaultdict, Counter


def load_trajectories_with_metadata(trajectories_path: str) -> Dict:
    """Load trajectories with metadata from JSON file."""
    with open(trajectories_path, 'r') as f:
        return json.load(f)


def build_graph_with_metadata(trajectories_data: Dict) -> nx.DiGraph:
    """
    Build directed graph from trajectories with metadata.
    
    Args:
        trajectories_data: Dictionary containing trajectories and cluster metadata
        
    Returns:
        NetworkX DiGraph with metadata
    """
    G = nx.DiGraph()
    
    trajectories = trajectories_data['trajectories']
    cluster_metadata = trajectories_data.get('cluster_metadata', {})
    
    # Track edges and collect all utterances per cluster
    edge_counts = defaultdict(int)
    edge_dialogues = defaultdict(set)
    cluster_utterances = defaultdict(list)  # Store all utterances with their source info
    
    # Process each trajectory
    for traj_data in trajectories:
        trajectory = traj_data['trajectory']
        dialogue_id = traj_data['dialogue_id']
        
        # Add nodes and edges
        for i in range(len(trajectory)):
            current_cluster = str(trajectory[i]['cluster_id'])
            
            # Collect utterance information for this cluster
            utterance_info = {
                'text': trajectory[i]['text'],
                'speaker': trajectory[i]['speaker'],
                'transcript_id': dialogue_id,
                'turn_idx': trajectory[i]['turn_idx']
            }
            cluster_utterances[current_cluster].append(utterance_info)
            
            # Add node if not exists (we'll update it later with all utterances)
            if current_cluster not in G.nodes:
                # Get metadata for this cluster
                metadata = cluster_metadata.get(current_cluster, {})
                
                G.add_node(
                    current_cluster,
                    label=f"C{current_cluster}",
                    **metadata  # Add all cluster metadata as node attributes
                )
            
            # Add edge to next cluster
            if i < len(trajectory) - 1:
                next_cluster = str(trajectory[i + 1]['cluster_id'])
                edge = (current_cluster, next_cluster)
                edge_counts[edge] += 1
                edge_dialogues[edge].add(dialogue_id)
    
    # Add all utterances and source information to each node
    for cluster_id, utterances in cluster_utterances.items():
        # Extract full text of all utterances
        utterance_texts = [u['text'] for u in utterances]
        
        # Extract source information (transcript_id and turn_idx)
        sources = [
            {
                'transcript_id': u['transcript_id'],
                'turn_idx': u['turn_idx'],
                'speaker': u['speaker']
            }
            for u in utterances
        ]
        
        # Update node with full utterance information
        G.nodes[cluster_id]['utterances'] = utterance_texts
        G.nodes[cluster_id]['utterance_sources'] = sources
        # Keep a representative utterance for backward compatibility
        G.nodes[cluster_id]['utterance'] = utterance_texts[0][:100] + '...' if len(utterance_texts[0]) > 100 else utterance_texts[0]
    
    # Add edges with weights
    for (source, target), count in edge_counts.items():
        G.add_edge(
            source,
            target,
            weight=count,
            num_dialogues=len(edge_dialogues[(source, target)])
        )
    
    return G


def export_graph_json(G: nx.DiGraph, output_path: str):
    """Export graph with metadata to JSON format."""
    # Convert graph to JSON-serializable format
    graph_data = {
        'nodes': [],
        'edges': []
    }
    
    # Export nodes with all metadata
    for node_id in G.nodes:
        node_data = {'id': node_id}
        node_data.update(G.nodes[node_id])
        graph_data['nodes'].append(node_data)
    
    # Export edges
    for source, target in G.edges:
        edge_data = {
            'source': source,
            'target': target
        }
        edge_data.update(G.edges[source, target])
        graph_data['edges'].append(edge_data)
    
    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"✓ Exported graph to JSON: {output_path}")


def export_graph_graphml(G: nx.DiGraph, output_path: str):
    """Export graph with metadata to GraphML format."""
    # Create a copy of the graph and convert list attributes to strings for GraphML compatibility
    G_copy = G.copy()
    
    print("\n[DEBUG] Checking node attributes for GraphML export...")
    for node_id in G_copy.nodes:
        for key, value in list(G_copy.nodes[node_id].items()):
            if value is None:
                print(f"[DEBUG] Node {node_id} has None value for key '{key}'")
            if isinstance(value, list):
                if key == 'utterances':
                    # Join utterances with newline for better readability
                    G_copy.nodes[node_id][key] = '\n---\n'.join(str(v) for v in value) if value else ''
                elif key == 'utterance_sources':
                    # Format sources as readable text
                    sources_text = '\n'.join(
                        f"- {s.get('transcript_id', 'N/A')} [turn {s.get('turn_idx', 'N/A')}] ({s.get('speaker', 'N/A')})"
                        for s in value
                    ) if value else ''
                    G_copy.nodes[node_id][key] = sources_text
                else:
                    # Convert other lists to comma-separated strings
                    G_copy.nodes[node_id][key] = ', '.join(str(v) for v in value) if value else ''
    
    nx.write_graphml(G_copy, output_path)
    print(f"✓ Exported graph to GraphML: {output_path}")


def export_graph_gexf(G: nx.DiGraph, output_path: str):
    """Export graph with metadata to GEXF format."""
    nx.write_gexf(G, output_path)
    print(f"✓ Exported graph to GEXF: {output_path}")


def generate_html_visualization(G: nx.DiGraph, output_path: str):
    """Generate interactive HTML visualization with metadata."""
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dialog2Flow Graph with Metadata</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }
        #graph {
            width: 100vw;
            height: 100vh;
        }
        .node {
            cursor: pointer;
        }
        .node circle {
            fill: #69b3a2;
            stroke: #333;
            stroke-width: 2px;
        }
        .node:hover circle {
            fill: #ff7f0e;
        }
        .node text {
            font-size: 12px;
            font-weight: bold;
            fill: #333;
            pointer-events: none;
        }
        .link {
            fill: none;
            stroke: #999;
            stroke-opacity: 0.6;
        }
        .link-label {
            font-size: 10px;
            fill: #666;
        }
        #tooltip {
            position: absolute;
            background: white;
            border: 2px solid #333;
            border-radius: 5px;
            padding: 10px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        #tooltip.visible {
            opacity: 1;
        }
        .metadata-item {
            margin: 5px 0;
            font-size: 12px;
        }
        .metadata-label {
            font-weight: bold;
            color: #555;
        }
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="tooltip"></div>
    
    <script>
        const graphData = GRAPH_DATA_PLACEHOLDER;
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Define arrow markers for directed edges
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 8)
            .attr("markerHeight", 8)
            .attr("xoverflow", "visible")
            .append("svg:path")
            .attr("d", "M 0,-5 L 10 ,0 L 0,5")
            .attr("fill", "#999")
            .style("stroke", "none");
        
        const g = svg.append("g");
        
        // Add zoom behavior
        svg.call(d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => g.attr("transform", event.transform)));
        
        // Create force simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.edges).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        // Create links
        const link = g.append("g")
            .selectAll("path")
            .data(graphData.edges)
            .enter().append("path")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.weight || 1))
            .attr("marker-end", "url(#arrowhead)");
        
        // Create nodes
        const node = g.append("g")
            .selectAll("g")
            .data(graphData.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        node.append("circle")
            .attr("r", d => 10 + (d.num_turns_in_cluster || 0) * 2);
        
        node.append("text")
            .attr("dx", 15)
            .attr("dy", 4)
            .text(d => d.label || d.id);
        
        // Tooltip
        const tooltip = d3.select("#tooltip");
        
        node.on("mouseover", function(event, d) {
            let html = `<div class="metadata-item"><span class="metadata-label">Cluster:</span> ${d.label || d.id}</div>`;
            html += `<div class="metadata-item"><span class="metadata-label">Turns in Cluster:</span> ${d.num_turns_in_cluster || 0}</div>`;
            
            // Display all utterances
            if (d.utterances && d.utterances.length > 0) {
                html += `<div class="metadata-item"><span class="metadata-label">Utterances (${d.utterances.length}):</span></div>`;
                d.utterances.forEach((utt, idx) => {
                    const preview = utt.length > 100 ? utt.substring(0, 100) + '...' : utt;
                    html += `<div class="metadata-item" style="margin-left: 10px; font-size: 11px;">• ${preview}</div>`;
                });
            }
            
            // Display sources (transcript_id and turn_idx)
            if (d.utterance_sources && d.utterance_sources.length > 0) {
                html += `<div class="metadata-item"><span class="metadata-label">Sources:</span></div>`;
                d.utterance_sources.forEach((src, idx) => {
                    html += `<div class="metadata-item" style="margin-left: 10px; font-size: 11px;">• ${src.transcript_id} [turn ${src.turn_idx}] (${src.speaker})</div>`;
                });
            }
            
            // Metadata scores
            if (d.escalation_level !== null && d.escalation_level !== undefined) {
                html += `<div class="metadata-item"><span class="metadata-label">Escalation:</span> ${d.escalation_level.toFixed(2)} (±${(d.escalation_level_std || 0).toFixed(2)})</div>`;
            }
            if (d.churn_risk_score !== null && d.churn_risk_score !== undefined) {
                html += `<div class="metadata-item"><span class="metadata-label">Churn Risk:</span> ${d.churn_risk_score.toFixed(2)} (±${(d.churn_risk_score_std || 0).toFixed(2)})</div>`;
            }
            if (d.empathy_score !== null && d.empathy_score !== undefined) {
                html += `<div class="metadata-item"><span class="metadata-label">Empathy:</span> ${d.empathy_score.toFixed(2)} (±${(d.empathy_score_std || 0).toFixed(2)})</div>`;
            }
            if (d.intents_emotions && d.intents_emotions.length > 0) {
                html += `<div class="metadata-item"><span class="metadata-label">Intents/Emotions:</span> ${d.intents_emotions.slice(0, 3).join(', ')}${d.intents_emotions.length > 3 ? '...' : ''}</div>`;
            }
            if (d.dialogue_acts && d.dialogue_acts.length > 0) {
                html += `<div class="metadata-item"><span class="metadata-label">Dialogue Acts:</span> ${d.dialogue_acts.slice(0, 2).join(', ')}${d.dialogue_acts.length > 2 ? '...' : ''}</div>`;
            }
            
            tooltip.html(html)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px")
                .classed("visible", true);
        })
        .on("mouseout", function() {
            tooltip.classed("visible", false);
        });
        
        // Update positions on tick
        simulation.on("tick", () => {
            link.attr("d", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                // Shorten the path to stop before the target node circle
                const targetRadius = 10 + (d.target.num_turns_in_cluster || 0) * 2;
                const offsetX = (dx * targetRadius) / dr;
                const offsetY = (dy * targetRadius) / dr;
                const targetX = d.target.x - offsetX;
                const targetY = d.target.y - offsetY;
                return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${targetX},${targetY}`;
            });
            
            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    </script>
</body>
</html>
"""
    
    # Export graph to JSON
    graph_json = {
        'nodes': [],
        'edges': []
    }
    
    for node_id in G.nodes:
        node_data = {'id': node_id}
        node_data.update(G.nodes[node_id])
        graph_json['nodes'].append(node_data)
    
    for source, target in G.edges:
        edge_data = {
            'source': source,
            'target': target
        }
        edge_data.update(G.edges[source, target])
        graph_json['edges'].append(edge_data)
    
    # Replace placeholder with actual data
    html_content = html_template.replace(
        'GRAPH_DATA_PLACEHOLDER',
        json.dumps(graph_json)
    )
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Generated interactive visualization: {output_path}")


def build_and_export_graph(
    trajectories_path: str,
    output_dir: str,
    formats: List[str] = ['json', 'graphml', 'html']
):
    """
    Build graph from trajectories and export in various formats.
    
    Args:
        trajectories_path: Path to trajectories JSON with metadata
        output_dir: Directory to save graph files
        formats: List of export formats ('json', 'graphml', 'gexf', 'html')
    """
    print(f"Loading trajectories from {trajectories_path}")
    trajectories_data = load_trajectories_with_metadata(trajectories_path)
    
    print("Building graph with metadata...")
    G = build_graph_with_metadata(trajectories_data)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export in requested formats
    if 'json' in formats:
        json_path = os.path.join(output_dir, 'graph_with_metadata.json')
        export_graph_json(G, json_path)
    
    if 'graphml' in formats:
        graphml_path = os.path.join(output_dir, 'graph_with_metadata.graphml')
        export_graph_graphml(G, graphml_path)
    
    if 'gexf' in formats:
        gexf_path = os.path.join(output_dir, 'graph_with_metadata.gexf')
        export_graph_gexf(G, gexf_path)
    
    if 'html' in formats:
        html_path = os.path.join(output_dir, 'graph_visualization.html')
        generate_html_visualization(G, html_path)
    
    print(f"✓ Graph with metadata exported to {output_dir}")
    
    return G


def main():
    parser = argparse.ArgumentParser(
        description='Build graph from trajectories with metadata'
    )
    parser.add_argument(
        '--trajectories',
        default='output/trajectories_with_metadata.json',
        help='Path to trajectories JSON with metadata'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Directory to save graph files'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['json', 'graphml', 'html'],
        choices=['json', 'graphml', 'gexf', 'html'],
        help='Export formats'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent

    trajectories_path = Path(args.trajectories)
    if not trajectories_path.is_absolute():
        trajectories_path = (base_dir / trajectories_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    build_and_export_graph(
        str(trajectories_path),
        str(output_dir),
        args.formats
    )


if __name__ == '__main__':
    main()
