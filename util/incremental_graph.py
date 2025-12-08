# -*- coding: utf-8 -*-
"""
Classes for incremental graph processing, caching, and augmentation.

Copyright (c) 2024 Idiap Research Institute
MIT License
"""
import time
import networkx as nx
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class IncrementalDialogStateCache:
    """
    Tier 1: Dialog State Tracking with Incremental Belief Updates
    """
    def __init__(self):
        self.previous_belief_state = None
        self.graph_cache = None
        self.context_cache = {}
    
    def extract_turn_belief(self, current_utterance, current_response):
        """Extract belief for current turn only (lightweight)"""
        # Placeholder for actual entity extraction logic
        # In a real system, this would use an NER model or similar
        turn_belief = {
            'utterance_entities': self._mock_extract_entities(current_utterance),
            'response_entities': self._mock_extract_entities(current_response)
        }
        return turn_belief
    
    def _mock_extract_entities(self, text):
        """Simple mock entity extraction for demonstration"""
        # Assume capitalized words are entities for now
        return [word.strip(".,!?") for word in text.split() if word[0].isupper()]

    def update_joint_belief(self, turn_belief):
        """Incrementally update, don't recompute everything"""
        if self.previous_belief_state is None:
            # First turn
            joint_belief = turn_belief
        else:
            # Integrate turn belief with previous state
            joint_belief = self.merge_beliefs(self.previous_belief_state, turn_belief)
        
        self.previous_belief_state = joint_belief
        return joint_belief
    
    def merge_beliefs(self, previous, current):
        """Merge previous and current beliefs intelligently"""
        merged = previous.copy()
        
        # Update/add new entities from current turn
        for key, entities in current.items():
            if key not in merged:
                merged[key] = []
            # Add unique new entities
            for entity in entities:
                if entity not in merged[key]:
                    merged[key].append(entity)
        
        return merged
    
    def cache_belief(self, question_id, graph_state=None):
        """Store belief state for future reuse"""
        self.context_cache[question_id] = {
            'belief_state': self.previous_belief_state,
            'graph': graph_state,
            'timestamp': time.time()
        }
    
    def get_cached_belief(self, question_id):
        return self.context_cache.get(question_id)


class IncrementalCausalGraph:
    """
    Tier 2: Incremental Graph Augmentation
    """
    def __init__(self, initial_graph):
        self.skeleton_graph = self.extract_skeleton(initial_graph)  # Upper layer
        self.subgraphs = self.partition_graph(initial_graph)  # Lower layer
        self.cached_states = {}
        self.full_graph = initial_graph
    
    def extract_skeleton(self, graph):
        """Create lightweight skeleton from main graph"""
        skeleton = nx.DiGraph()
        
        if not graph or len(graph.nodes()) == 0:
            return skeleton

        # Keep only high-degree nodes and critical connections
        degrees = [d for _, d in graph.degree()]
        if not degrees:
            return skeleton
            
        threshold = np.percentile(degrees, 75)
        high_degree_nodes = {node for node, deg in graph.degree() if deg > threshold}
        
        for node in high_degree_nodes:
            skeleton.add_node(node)
        
        # Add edges between high-degree nodes
        for node in high_degree_nodes:
            if node in graph:
                for neighbor in graph.neighbors(node):
                    if neighbor in high_degree_nodes:
                        skeleton.add_edge(node, neighbor, 
                                        weight=graph[node][neighbor].get('weight', 1))
        
        return skeleton
    
    def partition_graph(self, graph):
        """Partition graph into disjoint subgraphs"""
        subgraphs = {}
        
        # Use weakly connected components as a simple partitioning strategy
        # In a real large-scale graph, might use Metis or other partitioning algos
        for i, component in enumerate(nx.weakly_connected_components(graph)):
            subgraph_id = f"subgraph_{i}"
            subgraph = graph.subgraph(component).copy()
            subgraphs[subgraph_id] = {
                'graph': subgraph,
                'nodes': set(component),
                'cached_result': None,
                'is_modified': False
            }
        
        return subgraphs
    
    def augment_with_new_context(self, new_entities, new_relations):
        """Add new context without full reconstruction"""
        
        # print("[Step 1] Identifying affected subgraphs...")
        affected_subgraphs = self._identify_affected_subgraphs(new_entities)
        
        # print("[Step 2] Updating only affected subgraphs...")
        for subgraph_id in affected_subgraphs:
            self._update_subgraph(subgraph_id, new_entities, new_relations)
            self.subgraphs[subgraph_id]['is_modified'] = True
        
        # If new entities don't belong to any existing subgraph, create a new one
        unassigned_entities = [e for e in new_entities if not any(e in self.subgraphs[sg]['nodes'] for sg in self.subgraphs)]
        if unassigned_entities:
            new_subgraph_id = f"subgraph_{len(self.subgraphs)}"
            new_graph = nx.DiGraph()
            new_graph.add_nodes_from(unassigned_entities)
            # Add relations for these new entities
            for s, t, r in new_relations:
                if s in unassigned_entities or t in unassigned_entities:
                     new_graph.add_edge(s, t, relation=r)
            
            self.subgraphs[new_subgraph_id] = {
                'graph': new_graph,
                'nodes': set(unassigned_entities),
                'cached_result': None,
                'is_modified': True
            }
            affected_subgraphs.add(new_subgraph_id)

        # print("[Step 3] Propagating changes through skeleton...")
        self._propagate_skeleton_updates(affected_subgraphs)
        
        # Update full graph reference
        self._rebuild_full_graph_from_subgraphs()
        
        return affected_subgraphs
    
    def _rebuild_full_graph_from_subgraphs(self):
        """Reconstruct full graph from subgraphs (virtual view)"""
        self.full_graph = nx.DiGraph()
        for sg_info in self.subgraphs.values():
            self.full_graph = nx.compose(self.full_graph, sg_info['graph'])

    def _identify_affected_subgraphs(self, new_entities):
        """Find which subgraphs touch new entities"""
        affected = set()
        
        for entity in new_entities:
            for subgraph_id, subgraph_info in self.subgraphs.items():
                if entity in subgraph_info['nodes']:
                    affected.add(subgraph_id)
        
        # Also include skeleton-connected subgraphs (simplified logic)
        # In full implementation, check skeleton edges
        return affected
    
    def _update_subgraph(self, subgraph_id, new_entities, new_relations):
        """Update single subgraph with new context"""
        subgraph_info = self.subgraphs[subgraph_id]
        graph = subgraph_info['graph']
        
        # Add new entities that are relevant to this subgraph
        # (e.g. connected to existing nodes)
        relevant_new = []
        for entity in new_entities:
             # Check if connected to existing nodes in this subgraph via new_relations
             is_connected = False
             for s, t, _ in new_relations:
                 if (s == entity and t in subgraph_info['nodes']) or \
                    (t == entity and s in subgraph_info['nodes']):
                     is_connected = True
                     break
             if is_connected or entity in subgraph_info['nodes']:
                 relevant_new.append(entity)

        for entity in relevant_new:
            if entity not in graph:
                graph.add_node(entity)
                subgraph_info['nodes'].add(entity)
        
        # Add new relations
        for source, target, relation_type in new_relations:
            if source in subgraph_info['nodes'] and target in subgraph_info['nodes']:
                graph.add_edge(source, target, relation=relation_type)
        
        # Clear cached result
        subgraph_info['cached_result'] = None
    
    def _propagate_skeleton_updates(self, affected_subgraphs):
        """Update skeleton connections for affected subgraphs"""
        for subgraph_id in affected_subgraphs:
            subgraph = self.subgraphs[subgraph_id]['graph']
            
            # Update skeleton with any new high-degree nodes
            for node in subgraph.nodes():
                if subgraph.degree(node) > 3:  # Threshold
                    if node not in self.skeleton_graph:
                        self.skeleton_graph.add_node(node)


import pickle
import os
import torch

class SubgraphKVCache:
    """
    Tier 3: Subgraph-Level KV Caching (SubGCache)
    """
    def __init__(self, embedding_model=None, cache_dir="cache/kv"):
        self.embedding_model = embedding_model
        self.kv_cache = {}  # {subgraph_embedding_hash: kv_tensors}
        self.query_clusters = defaultdict(list)
        self.cache_dir = cache_dir
        self.metrics = {'hits': 0, 'misses': 0, 'saved_time': 0.0}
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def encode_subgraph(self, subgraph):
        """
        Create a deterministic structural hash/embedding for a subgraph.
        Uses Weisfeiler-Lehman Graph Hash for structural equality.
        """
        # WL hash is robust for isomorphism checks
        # We use 'label' attribute if available, otherwise node itself
        # Ensure node labels are strings for hashing
        
        # Create a view or copy with string labels for hashing if needed
        # But wl_hash expects a graph.
        # We need to ensure edge attributes are considered if important
        # For now, we focus on node labels and structure
        
        # Note: networkx.weisfeiler_lehman_graph_hash is available in newer nx versions
        # If not, we can implement a simple version or use another hash
        try:
            wl_hash = nx.weisfeiler_lehman_graph_hash(subgraph, node_attr='label')
        except (AttributeError, ImportError, KeyError):
            # Fallback if WL hash not available or 'label' missing
            # Simple structural hash: sorted edges + nodes
            nodes = sorted([str(n) for n in subgraph.nodes()])
            edges = sorted([(str(u), str(v)) for u, v in subgraph.edges()])
            wl_hash = hash(tuple(nodes + edges))
            
        return str(wl_hash)

    def cluster_queries_by_subgraph(self, queries, subgraphs):
        """Group similar queries by their retrieved subgraphs"""
        if not self.embedding_model:
            return []

        # print("Clustering queries by retrieved subgraph structure...")
        
        subgraph_embeddings = []
        for sg in subgraphs:
            # Embed subgraph structure - simplified: average of node embeddings or similar
            # Here assuming we have a way to encode subgraph
            # For now, we'll just use a placeholder random vector if no model
            embedding = np.random.rand(768) 
            subgraph_embeddings.append(embedding)
        
        if not subgraph_embeddings:
            return []

        # Cluster by similarity
        n_clusters = max(1, len(queries) // 5)  # Heuristic
        if len(queries) < 5:
            n_clusters = 1
            
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
        cluster_labels = kmeans.fit_predict(subgraph_embeddings)
        
        return cluster_labels
    
    def lookup_kv(self, subgraph):
        """Look up KV tensors for a subgraph"""
        subgraph_id = self.encode_subgraph(subgraph)
        kv = self.kv_cache.get(subgraph_id)
        if kv is not None:
            self.metrics['hits'] += 1
        else:
            self.metrics['misses'] += 1
        return kv

    def store_kv(self, subgraph, kv_tensors):
        """Store KV tensors for a subgraph"""
        subgraph_id = self.encode_subgraph(subgraph)
        self.kv_cache[subgraph_id] = kv_tensors
        self.save_cache()
        
    def cache_kv(self, subgraph, kv_tensors):
        """Alias for store_kv"""
        self.store_kv(subgraph, kv_tensors)

    def get_cached_kv(self, subgraph):
        """Alias for lookup_kv without metrics side effect (optional) or same"""
        return self.lookup_kv(subgraph)

    def simulate_reasoning(self, subgraph, compute_cost=0.5):
        """
        Simulate reasoning step with KV cache integration.
        Returns: (result, is_cached)
        """
        start_time = time.time()
        
        # 1. Check Cache
        cached_kv = self.lookup_kv(subgraph)
        
        if cached_kv is not None:
            # Hit! Return cached result (simulated)
            # In real scenario, we'd load KV into LLM and generate
            # Here we just return the cached tensor/value
            self.metrics['saved_time'] += compute_cost
            return cached_kv, True
        
        # 2. Miss - Compute
        time.sleep(compute_cost) # Simulate expensive computation
        
        # Generate result (mock)
        result = f"Reasoning result for {len(subgraph.nodes())} nodes"
        
        # 3. Store in Cache
        self.store_kv(subgraph, result)
        
        return result, False

    def store_kv_tensor(self, subgraph, kv_tensors: torch.Tensor):
        """Store KV tensor to disk as .pt file"""
        if kv_tensors is None:
            return
            
        subgraph_id = self.encode_subgraph(subgraph)
        tensor_path = os.path.join(self.cache_dir, f"{subgraph_id}.pt")
        try:
            torch.save(kv_tensors.cpu(), tensor_path)  # device-agnostic
            self.kv_cache[subgraph_id] = {"tensor_path": tensor_path, "hits": 0}
            self.save_cache()
        except Exception as e:
            print(f"Failed to store KV tensor: {e}")

    def load_kv_tensor(self, subgraph):
        """Load KV tensor from disk"""
        subgraph_id = self.encode_subgraph(subgraph)
        meta = self.kv_cache.get(subgraph_id)
        if meta is None or not isinstance(meta, dict) or "tensor_path" not in meta:
            return None
            
        try:
            tensor = torch.load(meta["tensor_path"])
            return tensor
        except Exception as e:
            print(f"Failed to load KV tensor: {e}")
            return None

    def run_reasoning_with_cache(self, subgraph, runtime, prompt):
        """
        Run reasoning using provided runtime, with KV caching.
        """
        # 1. Check Cache
        cached_tensor = self.load_kv_tensor(subgraph)

        # Cache HIT -> skip reasoning
        if cached_tensor is not None:
            self.metrics["hits"] += 1
            # Update hit count in metadata if we want
            subgraph_id = self.encode_subgraph(subgraph)
            if subgraph_id in self.kv_cache and isinstance(self.kv_cache[subgraph_id], dict):
                 self.kv_cache[subgraph_id]["hits"] += 1
            return f"[FAST-FWD] Cached reasoning result reused", True

        # Cache MISS -> call LLM
        output, tensor = runtime.run_with_optional_kv(prompt)
        
        # Store result (if tensor available)
        if tensor is not None:
            self.store_kv_tensor(subgraph, tensor)
            
        self.metrics["misses"] += 1
        return output, False

    def save_cache(self):
        """Persist cache to disk"""
        cache_file = os.path.join(self.cache_dir, "subgraph_kv.pkl")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.kv_cache, f)
        except Exception as e:
            print(f"Failed to save KV cache: {e}")

    def load_cache(self):
        """Load cache from disk"""
        cache_file = os.path.join(self.cache_dir, "subgraph_kv.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.kv_cache = pickle.load(f)
            except Exception as e:
                print(f"Failed to load KV cache: {e}")
