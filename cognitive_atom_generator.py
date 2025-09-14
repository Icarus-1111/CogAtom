#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import random
import time
import gc
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from tqdm import tqdm

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required. Install with: pip install networkx>=2.8.0")

# Global configuration constants
class CogAtomConfig:
    """Configuration parameters for cognitive atom generation."""
    
    # Core algorithm parameters
    COMBO_SIZE = 10                    # Target size for each knowledge combination
    N_STARTS = 3                       # Number of starting points for multi-start extension
    MAX_STEPS = 4                      # Maximum extension steps per starting point
    N_SAMPLES_PER_PATH = 10            # Number of combinations to generate per path
    
    # Probability and threshold settings
    PATH_EXTENSION_THRESHOLD = 0.3     # Minimum probability threshold for path extension
    BRIDGE_PROBABILITY_THRESHOLD = 0.5 # Threshold for weak connections requiring bridges
    PERTURB_PROB = 0.7                 # Probability of applying counterfactual perturbation
    
    # Dependency score to probability mapping
    SCORE_TO_PROB = {3: 0.3, 4: 0.6, 5: 0.85}
    
    # Performance and memory optimization
    BATCH_SIZE = 100                   # Number of paths to process in each batch
    FLUSH_THRESHOLD = 5000             # Combination count threshold for memory flush
    LOG_INTERVAL = 50                  # Interval for progress logging
    
    # Feature toggles
    BRIDGE_ENABLED = True              # Enable bridge replacement optimization
    PERTURBATION_ENABLED = True        # Enable counterfactual perturbation
    DETAILED_LOGGING = True            # Enable detailed operation logging
    ORDER_INDEPENDENT_DEDUP = True     # Enable order-independent deduplication
    
    # Quality control parameters
    MIN_CONNECTIVITY_SCORE = 0.1       # Minimum required connectivity score
    MAX_DUPLICATE_RATIO = 0.8          # Maximum allowed duplicate ratio

class DependencyGraph:
    """
    Manages the knowledge dependency graph with efficient operations for
    path extension and connectivity analysis.
    """
    
    def __init__(self, config: CogAtomConfig):
        """
        Initialize the dependency graph manager.
        
        Args:
            config: Configuration object containing algorithm parameters
        """
        self.config = config
        self.graph = nx.DiGraph()
        self.node_pool = set()
        self.probability_cache = {}
        
    def load_from_jsonl(self, dependency_file: str) -> bool:
        """
        Load dependency relationships from JSONL file and construct the graph.
        
        Args:
            dependency_file: Path to the dependency JSONL file
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        if not os.path.exists(dependency_file):
            print(f"[ERROR] Dependency file not found: {dependency_file}")
            return False
            
        try:
            with open(dependency_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        obj = json.loads(line.strip())
                        self._process_dependency_entry(obj)
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] Invalid JSON at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        print(f"[WARNING] Error processing line {line_num}: {e}")
                        continue
                        
            self._build_probability_cache()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load dependency file: {e}")
            return False
    
    def _process_dependency_entry(self, entry: Dict[str, Any]) -> None:
        """
        Process a single dependency entry and add to graph.
        
        Args:
            entry: Dictionary containing knowledge points and dependencies
        """
        knowledge_points = entry.get("knowledge_points", [])
        dependencies = entry.get("dependencies", [])
        
        # Add all knowledge points to the node pool
        self.node_pool.update(knowledge_points)
        
        # Process each dependency relationship
        for dep in dependencies:
            if not self._validate_dependency(dep):
                continue
                
            from_node = dep["from"]
            to_node = dep["to"]
            score = dep["score"]
            
            # Convert score to probability
            probability = self.config.SCORE_TO_PROB.get(score, 0.0)
            
            # Only add edges that meet the threshold
            if probability >= self.config.PATH_EXTENSION_THRESHOLD:
                self.graph.add_edge(from_node, to_node, 
                                  prob=probability, 
                                  score=score,
                                  reason=dep.get("reason", ""))
    
    def _validate_dependency(self, dep: Dict[str, Any]) -> bool:
        """
        Validate a dependency entry for required fields and values.
        
        Args:
            dep: Dependency dictionary to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = ["from", "to", "score"]
        if not all(field in dep for field in required_fields):
            return False
            
        if dep["score"] not in self.config.SCORE_TO_PROB:
            return False
            
        if dep["from"] == dep["to"]:
            return False
            
        return True
    
    def _build_probability_cache(self) -> None:
        """Build cache for efficient probability lookups."""
        self.probability_cache = {}
        for u, v, data in self.graph.edges(data=True):
            self.probability_cache[(u, v)] = data["prob"]
    
    def get_probability(self, from_node: str, to_node: str) -> float:
        """
        Get the dependency probability between two nodes.
        
        Args:
            from_node: Source node
            to_node: Target node
            
        Returns:
            float: Dependency probability (0.0 if no edge exists)
        """
        return self.probability_cache.get((from_node, to_node), 0.0)
    
    def get_successors(self, node: str) -> List[str]:
        """
        Get successor nodes with probabilities above threshold.
        
        Args:
            node: Node to get successors for
            
        Returns:
            List of successor nodes
        """
        if node not in self.graph:
            return []
        return list(self.graph.successors(node))
    
    def get_predecessors(self, node: str) -> List[str]:
        """
        Get predecessor nodes with probabilities above threshold.
        
        Args:
            node: Node to get predecessors for
            
        Returns:
            List of predecessor nodes
        """
        if node not in self.graph:
            return []
        return list(self.graph.predecessors(node))
    
    def find_bridge_nodes(self, from_node: str, to_node: str, 
                         excluded_nodes: Set[str]) -> List[Tuple[str, float]]:
        """
        Find potential bridge nodes between two nodes.
        
        Args:
            from_node: Source node
            to_node: Target node
            excluded_nodes: Nodes to exclude from bridge candidates
            
        Returns:
            List of (bridge_node, combined_probability) tuples
        """
        if from_node not in self.graph or to_node not in self.graph:
            return []
            
        # Find nodes that are successors of from_node and predecessors of to_node
        successors = set(self.get_successors(from_node))
        predecessors = set(self.get_predecessors(to_node))
        
        bridge_candidates = successors.intersection(predecessors)
        bridge_candidates = bridge_candidates - excluded_nodes
        
        # Calculate combined probabilities for each bridge
        bridges_with_prob = []
        for bridge in bridge_candidates:
            prob_1 = self.get_probability(from_node, bridge)
            prob_2 = self.get_probability(bridge, to_node)
            combined_prob = prob_1 * prob_2
            bridges_with_prob.append((bridge, combined_prob))
        
        # Sort by combined probability (descending)
        bridges_with_prob.sort(key=lambda x: x[1], reverse=True)
        return bridges_with_prob
    
    def calculate_connectivity_score(self, combination: List[str]) -> float:
        """
        Calculate the connectivity score for a knowledge combination.
        
        Args:
            combination: List of knowledge points
            
        Returns:
            float: Connectivity score between 0 and 1
        """
        if len(combination) < 2:
            return 1.0
            
        total_possible_edges = len(combination) * (len(combination) - 1)
        actual_connections = 0
        total_probability = 0.0
        
        for i, node_a in enumerate(combination):
            for j, node_b in enumerate(combination):
                if i != j:
                    prob = self.get_probability(node_a, node_b)
                    if prob > 0:
                        actual_connections += 1
                        total_probability += prob
        
        if actual_connections == 0:
            return 0.0
            
        # Normalize by both connection density and average probability
        density_score = actual_connections / total_possible_edges
        avg_probability = total_probability / actual_connections
        
        return (density_score + avg_probability) / 2.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dependency graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_pool_size": len(self.node_pool),
            "average_out_degree": sum(dict(self.graph.out_degree()).values()) / max(1, self.graph.number_of_nodes()),
            "average_in_degree": sum(dict(self.graph.in_degree()).values()) / max(1, self.graph.number_of_nodes()),
            "density": nx.density(self.graph),
            "strongly_connected_components": nx.number_strongly_connected_components(self.graph),
            "weakly_connected_components": nx.number_weakly_connected_components(self.graph)
        }

class PathExtender:
    """
    Implements multi-start path extension algorithm for generating
    knowledge combinations based on dependency relationships.
    """
    
    def __init__(self, dependency_graph: DependencyGraph, config: CogAtomConfig):
        """
        Initialize the path extender.
        
        Args:
            dependency_graph: Dependency graph manager
            config: Configuration object
        """
        self.graph = dependency_graph
        self.config = config
        
    def extend_path_multi_start(self, knowledge_points: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """
        Extend a path using multiple starting points with parallel expansion.
        
        Args:
            knowledge_points: Initial list of knowledge points from the path
            
        Returns:
            Tuple of (extended_combination, extension_trace)
        """
        if len(knowledge_points) < self.config.N_STARTS:
            n_starts = len(knowledge_points)
        else:
            n_starts = self.config.N_STARTS
            
        # Randomly select starting points
        start_points = random.sample(knowledge_points, n_starts)
        
        # Track global seen nodes to ensure uniqueness
        global_seen = set()
        all_paths = []
        extension_trace = {}
        
        # Extend from each starting point
        for start_point in start_points:
            path, steps_taken = self._extend_single_path(
                start_point, global_seen, self.config.MAX_STEPS
            )
            all_paths.append(path)
            extension_trace[start_point] = len(path)
            global_seen.update(path)
        
        # Merge all paths while maintaining uniqueness
        merged_combination = []
        seen_in_result = set()
        
        for path in all_paths:
            for node in path:
                if node not in seen_in_result:
                    merged_combination.append(node)
                    seen_in_result.add(node)
                    if len(merged_combination) >= self.config.COMBO_SIZE:
                        break
            if len(merged_combination) >= self.config.COMBO_SIZE:
                break
        
        return merged_combination[:self.config.COMBO_SIZE], extension_trace
    
    def _extend_single_path(self, start_node: str, global_seen: Set[str], 
                           max_steps: int) -> Tuple[List[str], int]:
        """
        Extend a single path from a starting node.
        
        Args:
            start_node: Node to start extension from
            global_seen: Set of globally seen nodes to avoid
            max_steps: Maximum number of extension steps
            
        Returns:
            Tuple of (path, steps_taken)
        """
        if start_node not in self.graph.graph:
            return [start_node], 0
            
        path = [start_node]
        current_node = start_node
        steps_taken = 0
        
        for step in range(max_steps):
            # Get candidate successors
            successors = self.graph.get_successors(current_node)
            
            # Filter out globally seen nodes and nodes already in current path
            candidates = []
            for succ in successors:
                if succ not in global_seen and succ not in path:
                    prob = self.graph.get_probability(current_node, succ)
                    if prob >= self.config.PATH_EXTENSION_THRESHOLD:
                        candidates.append((succ, prob))
            
            if not candidates:
                break
                
            # Select next node based on highest probability
            next_node = max(candidates, key=lambda x: x[1])[0]
            path.append(next_node)
            current_node = next_node
            steps_taken += 1
        
        return path, steps_taken

class BridgeReplacer:
    """
    Implements bridge replacement algorithm to optimize connectivity
    in knowledge combinations by inserting bridge nodes.
    """
    
    def __init__(self, dependency_graph: DependencyGraph, config: CogAtomConfig):
        """
        Initialize the bridge replacer.
        
        Args:
            dependency_graph: Dependency graph manager
            config: Configuration object
        """
        self.graph = dependency_graph
        self.config = config
        
    def optimize_combination(self, combination: List[str], 
                           log_operations: bool = False) -> Tuple[List[str], int]:
        """
        Optimize a combination by inserting bridge nodes for weak connections.
        
        Args:
            combination: Initial knowledge combination
            log_operations: Whether to log bridge operations
            
        Returns:
            Tuple of (optimized_combination, bridge_count)
        """
        if not self.config.BRIDGE_ENABLED:
            return combination, 0
            
        optimized = list(combination)
        bridge_count = 0
        i = 0
        
        while i < len(optimized) - 1:
            current_node = optimized[i]
            next_node = optimized[i + 1]
            
            # Check if connection is weak
            connection_prob = self.graph.get_probability(current_node, next_node)
            
            if connection_prob < self.config.BRIDGE_PROBABILITY_THRESHOLD:
                # Find potential bridge nodes
                excluded_nodes = set(optimized)
                bridges = self.graph.find_bridge_nodes(current_node, next_node, excluded_nodes)
                
                if bridges:
                    best_bridge, best_prob = bridges[0]
                    
                    if log_operations:
                        print(f"[BRIDGE] Weak connection ({current_node} -> {next_node}, "
                              f"prob={connection_prob:.3f})")
                        print(f"[BRIDGE] Inserting bridge node: {best_bridge} "
                              f"(combined_prob={best_prob:.3f})")
                    
                    # Insert bridge node
                    optimized.insert(i + 1, best_bridge)
                    bridge_count += 1
                    
                    # Remove duplicates and truncate to target size
                    optimized = self._deduplicate_and_truncate(optimized)
                    
                    # Skip the inserted bridge node
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return optimized, bridge_count
    
    def _deduplicate_and_truncate(self, combination: List[str]) -> List[str]:
        """
        Remove duplicates while preserving order and truncate to target size.
        
        Args:
            combination: Input combination with potential duplicates
            
        Returns:
            Deduplicated and truncated combination
        """
        seen = set()
        result = []
        
        for node in combination:
            if node not in seen:
                result.append(node)
                seen.add(node)
                if len(result) >= self.config.COMBO_SIZE:
                    break
        
        return result

class CounterfactualPerturber:
    """
    Implements counterfactual perturbation to enhance diversity
    in generated knowledge combinations.
    """
    
    def __init__(self, dependency_graph: DependencyGraph, config: CogAtomConfig):
        """
        Initialize the counterfactual perturber.
        
        Args:
            dependency_graph: Dependency graph manager
            config: Configuration object
        """
        self.graph = dependency_graph
        self.config = config
        
    def perturb_combination(self, combination: List[str], 
                          source_path: List[str]) -> List[str]:
        """
        Apply counterfactual perturbation to a combination.
        
        Args:
            combination: Input combination to perturb
            source_path: Original path for context-aware perturbation
            
        Returns:
            Perturbed combination
        """
        if not self.config.PERTURBATION_ENABLED:
            return combination
            
        if random.random() >= self.config.PERTURB_PROB:
            return combination
            
        perturbed = list(combination)
        used_nodes = set(perturbed)
        
        # First, try to extend from source path
        perturbed = self._extend_from_path(perturbed, source_path, used_nodes)
        
        # If still not enough, extend from global graph
        if len(perturbed) < self.config.COMBO_SIZE:
            perturbed = self._extend_from_graph(perturbed, used_nodes)
        
        return perturbed[:self.config.COMBO_SIZE]
    
    def _extend_from_path(self, combination: List[str], source_path: List[str], 
                         used_nodes: Set[str]) -> List[str]:
        """
        Extend combination using nodes from the source path.
        
        Args:
            combination: Current combination
            source_path: Source path for extension
            used_nodes: Set of already used nodes
            
        Returns:
            Extended combination
        """
        extended = list(combination)
        
        while len(extended) < self.config.COMBO_SIZE:
            # Select a random node from current combination as anchor
            if not extended:
                break
                
            anchor_node = random.choice(extended)
            
            # Find candidates from source path
            candidates = [node for node in source_path if node not in used_nodes]
            if not candidates:
                break
            
            # Select candidate with minimum dependency probability (counterfactual)
            best_candidate = None
            min_probability = float('inf')
            
            for candidate in candidates:
                prob = self.graph.get_probability(anchor_node, candidate)
                if prob < min_probability:
                    min_probability = prob
                    best_candidate = candidate
            
            if best_candidate:
                extended.append(best_candidate)
                used_nodes.add(best_candidate)
            else:
                break
        
        return extended
    
    def _extend_from_graph(self, combination: List[str], 
                          used_nodes: Set[str]) -> List[str]:
        """
        Extend combination using random nodes from the global graph.
        
        Args:
            combination: Current combination
            used_nodes: Set of already used nodes
            
        Returns:
            Extended combination
        """
        extended = list(combination)
        available_nodes = list(self.graph.node_pool - used_nodes)
        
        if not available_nodes:
            return extended
            
        # Shuffle for randomness
        random.shuffle(available_nodes)
        
        # Add nodes until target size is reached
        for node in available_nodes:
            if len(extended) >= self.config.COMBO_SIZE:
                break
            extended.append(node)
            used_nodes.add(node)
        
        return extended

class QualityAssessor:
    """
    Provides comprehensive quality assessment for generated combinations.
    """
    
    def __init__(self, dependency_graph: DependencyGraph, config: CogAtomConfig):
        """
        Initialize the quality assessor.
        
        Args:
            dependency_graph: Dependency graph manager
            config: Configuration object
        """
        self.graph = dependency_graph
        self.config = config
        
    def assess_combination(self, combination: List[str]) -> Dict[str, float]:
        """
        Assess the quality of a knowledge combination.
        
        Args:
            combination: Knowledge combination to assess
            
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['size'] = len(combination)
        metrics['uniqueness'] = len(set(combination)) / len(combination) if combination else 0
        
        # Connectivity metrics
        metrics['connectivity_score'] = self.graph.calculate_connectivity_score(combination)
        metrics['average_probability'] = self._calculate_average_probability(combination)
        metrics['dependency_density'] = self._calculate_dependency_density(combination)
        
        # Diversity metrics
        metrics['node_coverage'] = len(set(combination)) / len(self.graph.node_pool) if self.graph.node_pool else 0
        
        # Overall quality score (weighted combination)
        metrics['overall_quality'] = self._calculate_overall_quality(metrics)
        
        return metrics
    
    def _calculate_average_probability(self, combination: List[str]) -> float:
        """Calculate average dependency probability in combination."""
        if len(combination) < 2:
            return 0.0
            
        total_prob = 0.0
        connection_count = 0
        
        for i, node_a in enumerate(combination):
            for j, node_b in enumerate(combination):
                if i != j:
                    prob = self.graph.get_probability(node_a, node_b)
                    if prob > 0:
                        total_prob += prob
                        connection_count += 1
        
        return total_prob / connection_count if connection_count > 0 else 0.0
    
    def _calculate_dependency_density(self, combination: List[str]) -> float:
        """Calculate the density of dependency connections."""
        if len(combination) < 2:
            return 0.0
            
        total_possible = len(combination) * (len(combination) - 1)
        actual_connections = 0
        
        for i, node_a in enumerate(combination):
            for j, node_b in enumerate(combination):
                if i != j and self.graph.get_probability(node_a, node_b) > 0:
                    actual_connections += 1
        
        return actual_connections / total_possible
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'connectivity_score': 0.3,
            'average_probability': 0.25,
            'dependency_density': 0.25,
            'uniqueness': 0.1,
            'node_coverage': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            score += metrics.get(metric, 0.0) * weight
        
        return score
    
    def filter_high_quality(self, combinations: List[List[str]], 
                           min_quality: float = None) -> List[List[str]]:
        """
        Filter combinations based on quality thresholds.
        
        Args:
            combinations: List of combinations to filter
            min_quality: Minimum quality threshold (uses config default if None)
            
        Returns:
            Filtered list of high-quality combinations
        """
        if min_quality is None:
            min_quality = self.config.MIN_CONNECTIVITY_SCORE
            
        high_quality = []
        
        for combo in combinations:
            metrics = self.assess_combination(combo)
            if metrics['overall_quality'] >= min_quality:
                high_quality.append(combo)
        
        return high_quality

class OrderIndependentDeduplicator:
    """
    Handles order-independent deduplication of knowledge combinations.
    """
    
    def __init__(self, config: CogAtomConfig):
        """
        Initialize the deduplicator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    def normalize_combination(self, combination: List[str]) -> Tuple[str, ...]:
        """
        Normalize a combination for order-independent comparison.
        
        Args:
            combination: List of knowledge points
            
        Returns:
            Sorted tuple for order-independent deduplication
        """
        # Remove duplicates and sort alphabetically
        unique_nodes = list(set(combination))
        unique_nodes.sort()  # Lexicographic sorting for consistency
        return tuple(unique_nodes[:self.config.COMBO_SIZE])
    
    def add_combination(self, combinations: Set[Tuple[str, ...]], 
                       new_combo: List[str]) -> bool:
        """
        Add combination with order-independent deduplication.
        
        Args:
            combinations: Set of existing normalized combinations
            new_combo: New combination to add
            
        Returns:
            True if combination was added (not duplicate), False otherwise
        """
        if not self.config.ORDER_INDEPENDENT_DEDUP:
            # Fallback to original behavior
            if len(new_combo) == self.config.COMBO_SIZE:
                size_before = len(combinations)
                combinations.add(tuple(new_combo))
                return len(combinations) > size_before
            return False
        
        # Order-independent normalization
        normalized = self.normalize_combination(new_combo)
        if len(normalized) == self.config.COMBO_SIZE:
            size_before = len(combinations)
            combinations.add(normalized)
            return len(combinations) > size_before
        return False
    
    def deduplicate_combinations(self, combinations: List[List[str]]) -> List[List[str]]:
        """
        Deduplicate a list of combinations in an order-independent manner.
        
        Args:
            combinations: List of combinations to deduplicate
            
        Returns:
            List of unique combinations (order-independent)
        """
        if not self.config.ORDER_INDEPENDENT_DEDUP:
            # Fallback to set-based deduplication
            return list(set(tuple(combo) for combo in combinations))
        
        unique_normalized = set()
        result = []
        
        for combo in combinations:
            normalized = self.normalize_combination(combo)
            if normalized not in unique_normalized and len(normalized) == self.config.COMBO_SIZE:
                unique_normalized.add(normalized)
                result.append(list(normalized))  # Convert back to list
        
        return result

class CognitiveAtomGenerator:
    """
    Main generator class that orchestrates the cognitive atom combination
    generation process using dependency-driven algorithms with order-independent deduplication.
    """
    
    def __init__(self, config: Optional[CogAtomConfig] = None):
        """
        Initialize the cognitive atom generator.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or CogAtomConfig()
        self.dependency_graph = DependencyGraph(self.config)
        self.path_extender = None
        self.bridge_replacer = None
        self.perturber = None
        self.quality_assessor = None
        self.deduplicator = OrderIndependentDeduplicator(self.config)
        
        # Statistics tracking
        self.generation_stats = {
            'total_paths_processed': 0,
            'total_combinations_generated': 0,
            'unique_combinations': 0,
            'order_dependent_duplicates_removed': 0,
            'bridge_operations': 0,
            'perturbation_operations': 0,
            'quality_filtered': 0,
            'processing_time': 0.0
        }
        
        # Setup paths
        self.script_dir = Path(__file__).parent
        self.input_paths_file = (
            self.script_dir / "data" / "processed" / "random_walks" / 
            "cogatom_247_90_diverse" / "paths" / "diverse_random_walk_paths.txt"
        )
        self.dependency_file = (
            self.script_dir / "data" / "processed" / "dependencies" / 
            "cogatom_247_90_diverse" / "diverse_random_walk_dependencies.jsonl"
        )
        self.output_dir = (
            self.script_dir / "data" / "processed" / "combinations" / 
            "cogatom_247_90_diverse"
        )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = self.output_dir / "cognitive_atom_generation.log"
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Initialize logging system."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("CogAtom Cognitive Atom Generation Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {self.config.__dict__}\n\n")
    
    def _log(self, message: str, print_also: bool = False) -> None:
        """
        Write message to log file.
        
        Args:
            message: Message to log
            print_also: Whether to also print to console
        """
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
        
        if print_also or self.config.DETAILED_LOGGING:
            print(log_entry)
    
    def initialize(self) -> bool:
        """
        Initialize all components and load data.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        print("Initializing Cognitive Atom Generator...")
        self._log("Starting initialization process", True)
        
        # Validate input files
        if not self._validate_input_files():
            return False
        
        # Load dependency graph
        self._log("Loading dependency graph...", True)
        if not self.dependency_graph.load_from_jsonl(str(self.dependency_file)):
            self._log("Failed to load dependency graph", True)
            return False
        
        # Initialize algorithm components
        self.path_extender = PathExtender(self.dependency_graph, self.config)
        self.bridge_replacer = BridgeReplacer(self.dependency_graph, self.config)
        self.perturber = CounterfactualPerturber(self.dependency_graph, self.config)
        self.quality_assessor = QualityAssessor(self.dependency_graph, self.config)
        
        # Log graph statistics
        graph_stats = self.dependency_graph.get_statistics()
        self._log(f"Dependency graph loaded: {graph_stats}", True)
        
        # Log deduplication mode
        dedup_mode = "order-independent" if self.config.ORDER_INDEPENDENT_DEDUP else "order-dependent"
        self._log(f"Deduplication mode: {dedup_mode}", True)
        
        self._log("Initialization completed successfully", True)
        return True
    
    def _validate_input_files(self) -> bool:
        """
        Validate that all required input files exist.
        
        Returns:
            bool: True if all files exist, False otherwise
        """
        files_to_check = [
            ("Random walk paths", self.input_paths_file),
            ("Dependency relationships", self.dependency_file)
        ]
        
        for file_desc, file_path in files_to_check:
            if not file_path.exists():
                self._log(f"ERROR: {file_desc} file not found: {file_path}", True)
                return False
            self._log(f"✓ {file_desc} file found: {file_path}")
        
        return True
    
    def _parse_path_line(self, line: str) -> Optional[List[str]]:
        """
        Parse a single line from the random walk paths file.
        
        Args:
            line: Raw line from the paths file
            
        Returns:
            List of knowledge points or None if parsing fails
        """
        line = line.strip()
        if not line or not line.startswith("Diverse Random Walk from "):
            return None
        
        try:
            parts = line.split("##@@B")
            if len(parts) < 2:
                return None
            
            # Extract start node
            start_node = parts[0].replace("Diverse Random Walk from ", "").strip()
            
            # Extract path nodes
            path_part = parts[1]
            if len(parts) > 2:
                # Has additional information, take first part
                node_part = path_part
            else:
                # Check for additional markers
                if "##@@B" in path_part:
                    node_part = path_part.split("##@@B")[0]
                else:
                    node_part = path_part
            
            # Parse nodes
            nodes = [node.strip() for node in node_part.split("##") if node.strip()]
            
            # Combine start node with path nodes, ensuring uniqueness
            all_nodes = [start_node] + nodes
            unique_nodes = []
            seen = set()
            
            for node in all_nodes:
                if node not in seen:
                    seen.add(node)
                    unique_nodes.append(node)
            
            return unique_nodes if len(unique_nodes) >= 2 else None
            
        except Exception as e:
            self._log(f"Warning: Failed to parse line: {str(e)}")
            return None
    
    def _generate_combinations_for_path(self, knowledge_points: List[str], 
                                      path_index: int) -> Set[Tuple[str, ...]]:
        """
        Generate multiple combinations for a single knowledge path with order-independent deduplication.
        
        Args:
            knowledge_points: List of knowledge points from the path
            path_index: Index of the current path for logging
            
        Returns:
            Set of unique normalized combinations (as tuples)
        """
        combinations = set()
        
        # Log path information
        if len(knowledge_points) <= self.config.COMBO_SIZE:
            self._log(f"Path {path_index}: {len(knowledge_points)} points <= {self.config.COMBO_SIZE}, "
                     f"using all points")
            # Use all points if path is short, with normalization
            normalized_combo = self.deduplicator.normalize_combination(knowledge_points)
            if len(normalized_combo) > 0:
                combinations.add(normalized_combo)
            return combinations
        
        # Generate multiple combinations for longer paths
        for sample_idx in range(self.config.N_SAMPLES_PER_PATH):
            try:
                # Multi-start path extension
                extended_combo, extension_trace = self.path_extender.extend_path_multi_start(
                    knowledge_points
                )
                
                # Log extension details
                if sample_idx == 0:  # Log details for first sample
                    trace_summary = {k: v for k, v in extension_trace.items()}
                    self._log(f"Path {path_index}: Extended from {len(knowledge_points)} to "
                             f"{len(extended_combo)} points. Extension trace: {trace_summary}")
                
                # Apply counterfactual perturbation if needed
                if len(extended_combo) < self.config.COMBO_SIZE:
                    extended_combo = self.perturber.perturb_combination(
                        extended_combo, knowledge_points
                    )
                    self.generation_stats['perturbation_operations'] += 1
                
                # Bridge replacement optimization
                optimized_combo, bridge_count = self.bridge_replacer.optimize_combination(
                    extended_combo, log_operations=(sample_idx == 0)
                )
                
                if bridge_count > 0:
                    self.generation_stats['bridge_operations'] += bridge_count
                    if sample_idx == 0:
                        self._log(f"Path {path_index}: Applied {bridge_count} bridge operations")
                
                # Final perturbation
                final_combo = self.perturber.perturb_combination(optimized_combo, knowledge_points)
                
                # Ensure uniqueness and correct size
                unique_combo = []
                seen = set()
                for node in final_combo:
                    if node not in seen:
                        unique_combo.append(node)
                        seen.add(node)
                        if len(unique_combo) >= self.config.COMBO_SIZE:
                            break
                
                # Add with order-independent deduplication
                if len(unique_combo) == self.config.COMBO_SIZE:
                    was_added = self.deduplicator.add_combination(combinations, unique_combo)
                    if was_added:
                        self.generation_stats['total_combinations_generated'] += 1
                
            except Exception as e:
                self._log(f"Warning: Failed to generate combination for path {path_index}, "
                         f"sample {sample_idx}: {str(e)}")
                continue
        
        self._log(f"Path {path_index}: Generated {len(combinations)} unique combinations "
                 f"(order-independent deduplication: {self.config.ORDER_INDEPENDENT_DEDUP})")
        return combinations
    
    def generate_combinations(self) -> bool:
        """
        Generate cognitive atom combinations from all paths with order-independent deduplication.
        
        Returns:
            bool: True if generation successful, False otherwise
        """
        print("\nStarting cognitive atom combination generation...")
        self._log("Starting combination generation process", True)
        
        start_time = time.time()
        all_combinations = set()
        
        try:
            # Load and process paths
            with open(self.input_paths_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            total_lines = len(lines)
            self._log(f"Loaded {total_lines} paths for processing", True)
            
            # Process paths in batches
            batch_combinations = set()
            
            for i, line in enumerate(tqdm(lines, desc="Generating combinations")):
                knowledge_points = self._parse_path_line(line)
                if not knowledge_points:
                    continue
                
                # Generate combinations for this path
                path_combinations = self._generate_combinations_for_path(knowledge_points, i)
                batch_combinations.update(path_combinations)
                
                self.generation_stats['total_paths_processed'] += 1
                
                # Flush to memory periodically
                if len(batch_combinations) >= self.config.FLUSH_THRESHOLD:
                    all_combinations.update(batch_combinations)
                    batch_combinations.clear()
                    gc.collect()  # Force garbage collection
                    self._log(f"Memory flush at path {i}, total unique combinations: {len(all_combinations)}")
                
                # Progress logging
                if (i + 1) % self.config.LOG_INTERVAL == 0:
                    self._log(f"Processed {i + 1}/{total_lines} paths, "
                             f"current unique combinations: {len(all_combinations) + len(batch_combinations)}")
            
            # Final merge
            all_combinations.update(batch_combinations)
            
            # Update statistics
            self.generation_stats['unique_combinations'] = len(all_combinations)
            self.generation_stats['processing_time'] = time.time() - start_time
            
            # Calculate order-dependent duplicates removed
            if self.config.ORDER_INDEPENDENT_DEDUP:
                self.generation_stats['order_dependent_duplicates_removed'] = (
                    self.generation_stats['total_combinations_generated'] - 
                    self.generation_stats['unique_combinations']
                )
            
            # Save combinations
            self._save_combinations(all_combinations)
            
            # Generate quality analysis
            self._analyze_quality(all_combinations)
            
            self._log("Combination generation completed successfully", True)
            return True
            
        except Exception as e:
            self._log(f"ERROR: Combination generation failed: {str(e)}", True)
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _save_combinations(self, combinations: Set[Tuple[str, ...]]) -> None:
        """
        Save generated combinations to file.
        
        Args:
            combinations: Set of unique normalized combinations to save
        """
        output_file = self.output_dir / "cognitive_atom_combinations.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for combo in combinations:
                    f.write("##".join(combo) + "\n")
            
            self._log(f"Saved {len(combinations)} combinations to {output_file}", True)
            
        except Exception as e:
            self._log(f"ERROR: Failed to save combinations: {str(e)}", True)
    
    def _analyze_quality(self, combinations: Set[Tuple[str, ...]]) -> None:
        """
        Perform comprehensive quality analysis on generated combinations.
        
        Args:
            combinations: Set of combinations to analyze
        """
        self._log("Starting quality analysis...", True)
        
        if not combinations:
            self._log("No combinations to analyze", True)
            return
        
        # Sample combinations for detailed analysis (to avoid memory issues)
        sample_size = min(1000, len(combinations))
        sampled_combinations = random.sample(list(combinations), sample_size)
        
        quality_metrics = []
        connectivity_scores = []
        
        for combo in sampled_combinations:
            metrics = self.quality_assessor.assess_combination(list(combo))
            quality_metrics.append(metrics)
            connectivity_scores.append(metrics['connectivity_score'])
        
        # Calculate aggregate statistics
        avg_connectivity = sum(connectivity_scores) / len(connectivity_scores)
        avg_quality = sum(m['overall_quality'] for m in quality_metrics) / len(quality_metrics)
        
        # Knowledge point coverage analysis
        all_nodes_in_combos = set()
        for combo in sampled_combinations:
            all_nodes_in_combos.update(combo)
        
        coverage_ratio = len(all_nodes_in_combos) / len(self.dependency_graph.node_pool)
        
        # Save quality analysis
        quality_report = {
            'generation_statistics': self.generation_stats,
            'quality_metrics': {
                'average_connectivity_score': avg_connectivity,
                'average_overall_quality': avg_quality,
                'knowledge_point_coverage': coverage_ratio,
                'sample_size': sample_size,
                'total_combinations': len(combinations)
            },
            'deduplication_info': {
                'order_independent_enabled': self.config.ORDER_INDEPENDENT_DEDUP,
                'order_dependent_duplicates_removed': self.generation_stats.get('order_dependent_duplicates_removed', 0)
            },
            'configuration': self.config.__dict__,
            'dependency_graph_stats': self.dependency_graph.get_statistics()
        }
        
        # Save to JSON
        stats_file = self.output_dir / "generation_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        report_file = self.output_dir / "quality_analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CogAtom Cognitive Atom Generation Quality Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Generation Statistics:\n")
            f.write("-" * 30 + "\n")
            for key, value in self.generation_stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nDeduplication Information:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Order-independent deduplication: {self.config.ORDER_INDEPENDENT_DEDUP}\n")
            if self.config.ORDER_INDEPENDENT_DEDUP:
                f.write(f"Order-dependent duplicates removed: {self.generation_stats.get('order_dependent_duplicates_removed', 0)}\n")
            
            f.write(f"\nQuality Metrics (sample of {sample_size} combinations):\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average connectivity score: {avg_connectivity:.3f}\n")
            f.write(f"Average overall quality: {avg_quality:.3f}\n")
            f.write(f"Knowledge point coverage: {coverage_ratio:.3f}\n")
            
            f.write(f"\nConfiguration:\n")
            f.write("-" * 30 + "\n")
            for key, value in self.config.__dict__.items():
                f.write(f"{key}: {value}\n")
        
        self._log(f"Quality analysis completed. Average connectivity: {avg_connectivity:.3f}, "
                 f"Average quality: {avg_quality:.3f}, Coverage: {coverage_ratio:.3f}", True)
    
    def run(self) -> bool:
        """
        Run the complete cognitive atom generation pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("CogAtom Cognitive Atom Generator")
        print("=" * 50)
        
        # Initialize system
        if not self.initialize():
            print("❌ Initialization failed")
            return False
        
        # Generate combinations
        if not self.generate_combinations():
            print("❌ Combination generation failed")
            return False
        
        # Print final summary
        print("\n" + "=" * 50)
        print("Generation Summary:")
        print(f"  Paths processed: {self.generation_stats['total_paths_processed']}")
        print(f"  Combinations generated: {self.generation_stats['total_combinations_generated']}")
        print(f"  Unique combinations: {self.generation_stats['unique_combinations']}")
        if self.config.ORDER_INDEPENDENT_DEDUP:
            print(f"  Order-dependent duplicates removed: {self.generation_stats.get('order_dependent_duplicates_removed', 0)}")
        print(f"  Bridge operations: {self.generation_stats['bridge_operations']}")
        print(f"  Perturbation operations: {self.generation_stats['perturbation_operations']}")
        print(f"  Processing time: {self.generation_stats['processing_time']:.1f} seconds")
        print(f"  Deduplication mode: {'Order-independent' if self.config.ORDER_INDEPENDENT_DEDUP else 'Order-dependent'}")
        print(f"\nOutput files saved to: {self.output_dir}")
        print("✅ Cognitive atom generation completed successfully!")
        
        return True

def main():
    """Main execution function."""
    # Create generator with default configuration
    generator = CognitiveAtomGenerator()
    
    # Run the generation pipeline
    success = generator.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
