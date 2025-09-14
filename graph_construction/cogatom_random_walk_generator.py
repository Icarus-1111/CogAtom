#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CogAtom Diverse Random Walk Generator

"""

import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

import networkx as nx
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('random_walk_generation.log')
    ]
)
logger = logging.getLogger(__name__)


class WalkConfig:
    """Configuration class for random walk parameters."""
    
    # Core algorithm parameters
    N_HOPS = 4                      # N-hop neighbor expansion depth
    MAX_NODES = 15                  # Target path length
    PATH_COUNT = 1                  # Number of paths per starting node
    USE_DEGREE_PENALTY = True       # Enable degree penalty for diversity
    DEGREE_PENALTY_BETA = 1.0       # Degree penalty strength
    USE_DISTURBANCE = True          # Enable probability disturbance
    DISTURBANCE_LEVEL = 0.1         # Disturbance strength
    MINIMUM_COMPOSITE_NODES = 6     # Minimum acceptable path length
    EPSILON = 1e-6                  # Numerical stability parameter
    
    # Quality control parameters
    MAX_RETRY_ATTEMPTS = 3          # Maximum retry attempts for failed paths
    MIN_PATH_DIVERSITY = 0.3        # Minimum required path diversity score
    
    # Performance parameters
    BATCH_SIZE = 1000               # Batch size for processing nodes
    MEMORY_LIMIT_MB = 2048          # Memory limit in MB


class CogAtomRandomWalkGenerator:
    """
    Main class for generating diverse random walk paths from CogAtom knowledge graph.
    
    This class implements the complete pipeline from graph loading to path generation,
    including diversity sampling, quality analysis, and comprehensive reporting.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize the random walk generator.
        
        Args:
            config_override: Optional dictionary to override default configuration.
        """
        # Setup paths using relative path resolution
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.data_dir = self.project_root / "data"
        
        # Input paths
        self.input_graph_path = (
            self.data_dir / "processed" / "knowledge_graphs" / 
            "cogatom_247_90" / "cleaned" / "knowledge_graph_cleaned.gexf"
        )
        self.input_stats_path = (
            self.data_dir / "processed" / "knowledge_graphs" / 
            "cogatom_247_90" / "analysis" / "graph_statistics.json"
        )
        
        # Output paths
        self.output_base = self.data_dir / "processed" / "random_walks"
        self.output_dir = self.output_base / "cogatom_247_90_diverse"
        self.paths_dir = self.output_dir / "paths"
        self.analysis_dir = self.output_dir / "analysis"
        self.logs_dir = self.output_dir / "logs"
        self.config_dir = self.output_dir / "config"
        
        # Create output directories
        for dir_path in [self.paths_dir, self.analysis_dir, self.logs_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Apply configuration overrides
        if config_override:
            for key, value in config_override.items():
                if hasattr(WalkConfig, key.upper()):
                    setattr(WalkConfig, key.upper(), value)
        
        # Data containers
        self.graph: Optional[nx.Graph] = None
        self.graph_stats: Dict[str, Any] = {}
        self.all_paths: List[Tuple[str, List[str], List[str]]] = []
        self.generation_stats: Dict[str, Any] = {}
        self.failure_analysis: Dict[str, Any] = {}
        
        logger.info(f"CogAtom Random Walk Generator initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def validate_input_files(self) -> bool:
        """
        Validate that all required input files exist and are readable.
        
        Returns:
            bool: True if all files are valid, False otherwise.
        """
        logger.info("Validating input files...")
        
        required_files = [
            ("Knowledge graph file", self.input_graph_path),
            ("Graph statistics file", self.input_stats_path)
        ]
        
        for file_desc, file_path in required_files:
            if not file_path.exists():
                logger.error(f"{file_desc} not found: {file_path}")
                return False
            
            if not file_path.is_file():
                logger.error(f"{file_desc} is not a file: {file_path}")
                return False
            
            logger.info(f"✓ {file_desc}: {file_path}")
        
        return True
    
    def load_knowledge_graph(self) -> bool:
        """
        Load the cleaned knowledge graph from GEXF file.
        
        Returns:
            bool: True if loading successful, False otherwise.
        """
        logger.info(f"Loading knowledge graph from: {self.input_graph_path}")
        
        try:
            self.graph = nx.read_gexf(str(self.input_graph_path))
            
            # Validate graph structure
            if self.graph.number_of_nodes() == 0:
                logger.error("Loaded graph has no nodes")
                return False
            
            if self.graph.number_of_edges() == 0:
                logger.error("Loaded graph has no edges")
                return False
            
            # Check for weight attributes
            sample_edges = list(self.graph.edges(data=True))[:5]
            has_weights = all('weight' in edge_data for _, _, edge_data in sample_edges)
            
            if not has_weights:
                logger.warning("Graph edges may not have weight attributes")
            
            logger.info(f"Graph loaded successfully:")
            logger.info(f"  Nodes: {self.graph.number_of_nodes()}")
            logger.info(f"  Edges: {self.graph.number_of_edges()}")
            logger.info(f"  Average degree: {2 * self.graph.number_of_edges() / self.graph.number_of_nodes():.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            return False
    
    def load_graph_statistics(self) -> bool:
        """
        Load graph statistics from JSON file.
        
        Returns:
            bool: True if loading successful, False otherwise.
        """
        logger.info(f"Loading graph statistics from: {self.input_stats_path}")
        
        try:
            with open(self.input_stats_path, 'r', encoding='utf-8') as f:
                self.graph_stats = json.load(f)
            
            logger.info("Graph statistics loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load graph statistics: {e}")
            return False
    
    def apply_noise_to_probability(self, probabilities: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Apply noise to sampling probabilities to enhance path diversity.
        
        Args:
            probabilities: Original probability distribution
            noise_level: Noise strength (0.0 to 1.0)
            
        Returns:
            np.ndarray: Disturbed probability distribution
        """
        if noise_level <= 0:
            return probabilities
        
        noise = noise_level * (np.random.random(probabilities.shape) - 0.5)
        disturbed_probabilities = probabilities + noise
        disturbed_probabilities = np.maximum(disturbed_probabilities, 0)
        
        # Renormalize
        prob_sum = np.sum(disturbed_probabilities)
        if prob_sum > 0:
            disturbed_probabilities /= prob_sum
        else:
            disturbed_probabilities = np.ones(len(probabilities)) / len(probabilities)
        
        return disturbed_probabilities
    
    def get_all_nth_neighbors(self, start_node: str, n: int) -> Set[str]:
        """
        Get all n-hop neighbors of a starting node.
        
        Args:
            start_node: Starting node
            n: Number of hops
            
        Returns:
            Set[str]: Set of all n-hop neighbors including the start node
        """
        current_neighbors = {start_node}
        all_neighbors = set(current_neighbors)
        
        for _ in range(n):
            next_neighbors = set()
            for node in current_neighbors:
                if node in self.graph:
                    neighbors = set(self.graph.neighbors(node))
                    next_neighbors.update(neighbors)
            all_neighbors.update(next_neighbors)
            current_neighbors = next_neighbors
            
            if not next_neighbors:  # No more neighbors to expand
                break
        
        return all_neighbors
    
    def compute_diverse_weights(self, current_node: str, neighbors: List[str], 
                              beta: float = 1.0, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute diversity sampling weights with degree penalty.
        
        Formula: weight = edge_weight / (log(1 + degree)^beta + epsilon)
        
        Args:
            current_node: Current node in the walk
            neighbors: List of neighbor nodes
            beta: Degree penalty strength
            epsilon: Numerical stability parameter
            
        Returns:
            np.ndarray: Diversity-adjusted weights
        """
        weights = []
        
        for neighbor in neighbors:
            # Get edge weight
            edge_data = self.graph[current_node].get(neighbor, {})
            edge_weight = float(edge_data.get('weight', 1.0))
            
            # Get neighbor degree
            neighbor_degree = self.graph.degree(neighbor)
            
            # Compute diversity weight
            degree_penalty = np.log1p(neighbor_degree) ** beta + epsilon
            diverse_weight = edge_weight / degree_penalty
            
            weights.append(diverse_weight)
        
        return np.array(weights)
    
    def generate_single_random_walk(self, start_node: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Generate a single diverse random walk path from a starting node.
        
        Args:
            start_node: Starting node for the walk
            
        Returns:
            Tuple containing:
            - List[str]: Path nodes
            - List[str]: Step weights
            - Dict[str, Any]: Generation metadata
        """
        metadata = {
            'start_node': start_node,
            'success': False,
            'failure_reason': None,
            'path_length': 0,
            'n_hop_neighbors': 0,
            'backtrack_count': 0
        }
        
        # Get n-hop neighbors
        nth_neighbors = self.get_all_nth_neighbors(start_node, WalkConfig.N_HOPS)
        num_nth_neighbors = len(nth_neighbors)
        metadata['n_hop_neighbors'] = num_nth_neighbors
        
        # Check if we have enough neighbors
        if num_nth_neighbors < WalkConfig.MAX_NODES:
            if num_nth_neighbors >= WalkConfig.MINIMUM_COMPOSITE_NODES:
                # Use all available neighbors
                selected_nodes = list(nth_neighbors)
                weight_details = self._compute_path_weights(selected_nodes)
                metadata['success'] = True
                metadata['path_length'] = len(selected_nodes)
                return selected_nodes, weight_details, metadata
            else:
                metadata['failure_reason'] = f"Insufficient neighbors: {num_nth_neighbors} < {WalkConfig.MINIMUM_COMPOSITE_NODES}"
                return [], [], metadata
        
        # Initialize walk
        current_node = start_node
        selected_nodes = {current_node}
        weight_details = []
        history_stack = [(current_node, list(set(self.graph.neighbors(current_node)) - selected_nodes))]
        
        # Perform random walk with backtracking
        while len(selected_nodes) < WalkConfig.MAX_NODES:
            if not history_stack:
                break
            
            current_node, neighbors = history_stack[-1]
            
            # If no more neighbors, backtrack
            if not neighbors:
                history_stack.pop()
                metadata['backtrack_count'] += 1
                continue
            
            # Compute sampling weights
            if WalkConfig.USE_DEGREE_PENALTY:
                weights = self.compute_diverse_weights(
                    current_node, neighbors, 
                    beta=WalkConfig.DEGREE_PENALTY_BETA, 
                    epsilon=WalkConfig.EPSILON
                )
            else:
                weights = np.array([
                    float(self.graph[current_node].get(neighbor, {}).get('weight', 1.0))
                    for neighbor in neighbors
                ])
            
            # Normalize to probabilities
            if np.sum(weights) == 0:
                probabilities = np.ones(len(weights)) / len(weights)
            else:
                probabilities = weights / np.sum(weights)
            
            # Apply disturbance if enabled
            if WalkConfig.USE_DISTURBANCE and WalkConfig.DISTURBANCE_LEVEL > 0:
                probabilities = self.apply_noise_to_probability(
                    probabilities, WalkConfig.DISTURBANCE_LEVEL
                )
            
            # Sample next node
            next_node = np.random.choice(neighbors, p=probabilities)
            neighbors.remove(next_node)
            history_stack[-1] = (current_node, neighbors)
            
            # Add node to path if not already selected
            if next_node not in selected_nodes:
                selected_nodes.add(next_node)
                
                # Record edge weight
                edge_data = self.graph[current_node].get(next_node, {})
                edge_weight = edge_data.get('weight', 1.0)
                weight_details.append(str(edge_weight))
                
                # Add to history stack
                next_neighbors = list(set(self.graph.neighbors(next_node)) - selected_nodes)
                history_stack.append((next_node, next_neighbors))
        
        metadata['success'] = True
        metadata['path_length'] = len(selected_nodes)
        
        return list(selected_nodes), weight_details, metadata
    
    def _compute_path_weights(self, path_nodes: List[str]) -> List[str]:
        """
        Compute weights for a given path.
        
        Args:
            path_nodes: List of nodes in the path
            
        Returns:
            List[str]: String representations of edge weights
        """
        weight_details = []
        
        for i in range(len(path_nodes) - 1):
            current = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            if self.graph.has_edge(current, next_node):
                edge_data = self.graph[current][next_node]
                weight = edge_data.get('weight', 1.0)
                weight_details.append(str(weight))
            else:
                weight_details.append('0.0')  # No direct edge
        
        return weight_details
    
    def generate_all_random_walks(self) -> bool:
        """
        Generate random walks for all nodes in the graph.
        
        Returns:
            bool: True if generation successful, False otherwise.
        """
        logger.info("Starting random walk generation for all nodes...")
        
        nodes = list(self.graph.nodes())
        total_nodes = len(nodes)
        
        # Initialize statistics
        self.generation_stats = {
            'total_nodes': total_nodes,
            'successful_starts': 0,
            'failed_starts': 0,
            'total_paths_generated': 0,
            'average_path_length': 0.0,
            'success_rate': 0.0,
            'processing_time': 0.0
        }
        
        self.failure_analysis = {
            'insufficient_neighbors': 0,
            'connectivity_issues': 0,
            'other_failures': 0,
            'failure_reasons': defaultdict(list)
        }
        
        start_time = time.time()
        
        # Generate paths for all nodes
        successful_paths = []
        failed_nodes = []
        path_lengths = []
        
        for node in tqdm(nodes, desc="Generating diverse random walks"):
            for _ in range(WalkConfig.PATH_COUNT):
                path_nodes, weight_details, metadata = self.generate_single_random_walk(node)
                
                if metadata['success']:
                    successful_paths.append((node, path_nodes, weight_details))
                    path_lengths.append(metadata['path_length'])
                    self.generation_stats['successful_starts'] += 1
                    self.generation_stats['total_paths_generated'] += 1
                else:
                    failed_nodes.append((node, metadata['failure_reason']))
                    self.generation_stats['failed_starts'] += 1
                    
                    # Categorize failure
                    if 'Insufficient neighbors' in metadata['failure_reason']:
                        self.failure_analysis['insufficient_neighbors'] += 1
                    else:
                        self.failure_analysis['other_failures'] += 1
                    
                    self.failure_analysis['failure_reasons'][metadata['failure_reason']].append(node)
        
        # Update statistics
        end_time = time.time()
        self.generation_stats['processing_time'] = end_time - start_time
        
        if self.generation_stats['total_paths_generated'] > 0:
            self.generation_stats['average_path_length'] = np.mean(path_lengths)
        
        if total_nodes > 0:
            self.generation_stats['success_rate'] = self.generation_stats['successful_starts'] / total_nodes
        
        self.all_paths = successful_paths
        
        logger.info(f"Random walk generation completed:")
        logger.info(f"  Total nodes: {total_nodes}")
        logger.info(f"  Successful paths: {self.generation_stats['total_paths_generated']}")
        logger.info(f"  Failed attempts: {self.generation_stats['failed_starts']}")
        logger.info(f"  Success rate: {self.generation_stats['success_rate']:.2%}")
        logger.info(f"  Average path length: {self.generation_stats['average_path_length']:.2f}")
        logger.info(f"  Processing time: {self.generation_stats['processing_time']:.2f} seconds")
        
        return len(successful_paths) > 0
    
    def analyze_path_quality(self) -> Dict[str, Any]:
        """
        Analyze the quality and diversity of generated paths.
        
        Returns:
            Dict[str, Any]: Comprehensive path quality analysis
        """
        logger.info("Analyzing path quality and diversity...")
        
        if not self.all_paths:
            logger.warning("No paths available for analysis")
            return {}
        
        analysis = {
            'path_statistics': {},
            'node_coverage': {},
            'diversity_metrics': {},
            'weight_analysis': {}
        }
        
        # Path length statistics
        path_lengths = [len(path_nodes) for _, path_nodes, _ in self.all_paths]
        length_distribution = Counter(path_lengths)
        
        analysis['path_statistics'] = {
            'total_paths': len(self.all_paths),
            'length_distribution': dict(length_distribution),
            'min_length': min(path_lengths) if path_lengths else 0,
            'max_length': max(path_lengths) if path_lengths else 0,
            'mean_length': np.mean(path_lengths) if path_lengths else 0,
            'std_length': np.std(path_lengths) if path_lengths else 0
        }
        
        # Node coverage analysis
        all_nodes_in_paths = set()
        node_frequency = Counter()
        
        for _, path_nodes, _ in self.all_paths:
            all_nodes_in_paths.update(path_nodes)
            node_frequency.update(path_nodes)
        
        total_graph_nodes = self.graph.number_of_nodes()
        coverage_rate = len(all_nodes_in_paths) / total_graph_nodes if total_graph_nodes > 0 else 0
        
        analysis['node_coverage'] = {
            'nodes_in_paths': len(all_nodes_in_paths),
            'total_graph_nodes': total_graph_nodes,
            'coverage_rate': coverage_rate,
            'most_frequent_nodes': node_frequency.most_common(10),
            'unique_nodes_per_path': len(all_nodes_in_paths) / len(self.all_paths) if self.all_paths else 0
        }
        
        # Diversity metrics
        if len(self.all_paths) > 1:
            # Calculate pairwise Jaccard similarity
            similarities = []
            paths_as_sets = [set(path_nodes) for _, path_nodes, _ in self.all_paths]
            
            for i in range(len(paths_as_sets)):
                for j in range(i + 1, len(paths_as_sets)):
                    set_i, set_j = paths_as_sets[i], paths_as_sets[j]
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    jaccard = intersection / union if union > 0 else 0
                    similarities.append(jaccard)
            
            analysis['diversity_metrics'] = {
                'average_jaccard_similarity': np.mean(similarities) if similarities else 0,
                'diversity_score': 1 - np.mean(similarities) if similarities else 1,
                'similarity_std': np.std(similarities) if similarities else 0
            }
        
        # Weight analysis
        all_weights = []
        for _, _, weight_details in self.all_paths:
            try:
                weights = [float(w) for w in weight_details if w != '0.0']
                all_weights.extend(weights)
            except ValueError:
                continue
        
        if all_weights:
            analysis['weight_analysis'] = {
                'total_edges': len(all_weights),
                'mean_weight': np.mean(all_weights),
                'std_weight': np.std(all_weights),
                'min_weight': min(all_weights),
                'max_weight': max(all_weights),
                'weight_distribution': dict(Counter([round(w, 2) for w in all_weights]))
            }
        
        logger.info("Path quality analysis completed")
        return analysis
    
    def save_paths_to_file(self) -> str:
        """
        Save generated paths to text file.
        
        Returns:
            str: Path to the saved file
        """
        logger.info("Saving paths to file...")
        
        output_file = self.paths_dir / "diverse_random_walk_paths.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for start_node, path_nodes, weight_details in self.all_paths:
                    path_str = '##'.join(path_nodes)
                    weights_str = ' -> '.join(weight_details) if weight_details else 'N/A'
                    
                    line = f"Diverse Random Walk from {start_node}##@@B{path_str}##@@BStep Weights: {weights_str}\n"
                    f.write(line)
            
            logger.info(f"Paths saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save paths: {e}")
            return ""
    
    def save_statistics(self, quality_analysis: Dict[str, Any]) -> str:
        """
        Save comprehensive statistics to JSON file.
        
        Args:
            quality_analysis: Path quality analysis results
            
        Returns:
            str: Path to the saved statistics file
        """
        logger.info("Saving statistics to file...")
        
        stats_file = self.paths_dir / "path_statistics.json"
        
        comprehensive_stats = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'project': 'CogAtom',
                'algorithm': 'Diverse (Degree-Penalized) Random Walk',
                'version': '1.0.0'
            },
            'configuration': {
                'n_hops': WalkConfig.N_HOPS,
                'max_nodes': WalkConfig.MAX_NODES,
                'path_count': WalkConfig.PATH_COUNT,
                'use_degree_penalty': WalkConfig.USE_DEGREE_PENALTY,
                'degree_penalty_beta': WalkConfig.DEGREE_PENALTY_BETA,
                'use_disturbance': WalkConfig.USE_DISTURBANCE,
                'disturbance_level': WalkConfig.DISTURBANCE_LEVEL,
                'minimum_composite_nodes': WalkConfig.MINIMUM_COMPOSITE_NODES
            },
            'generation_statistics': self.generation_stats,
            'failure_analysis': {
                'insufficient_neighbors': self.failure_analysis['insufficient_neighbors'],
                'connectivity_issues': self.failure_analysis['connectivity_issues'],
                'other_failures': self.failure_analysis['other_failures'],
                'failure_reasons_summary': {
                    reason: len(nodes) 
                    for reason, nodes in self.failure_analysis['failure_reasons'].items()
                }
            },
            'quality_analysis': quality_analysis
        }
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Statistics saved to: {stats_file}")
            return str(stats_file)
            
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
            return ""
    
    def generate_comprehensive_report(self, quality_analysis: Dict[str, Any]) -> str:
        """
        Generate a comprehensive text report.
        
        Args:
            quality_analysis: Path quality analysis results
            
        Returns:
            str: Path to the generated report file
        """
        logger.info("Generating comprehensive report...")
        
        report_file = self.analysis_dir / "walk_generation_report.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("CogAtom Diverse Random Walk Generation Report\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Generation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Algorithm: Diverse (Degree-Penalized) Random Walk\n")
                f.write(f"Project: CogAtom Knowledge Graph\n\n")
                
                # Configuration
                f.write("Algorithm Configuration:\n")
                f.write("-" * 30 + "\n")
                f.write(f"N-hop neighbors: {WalkConfig.N_HOPS}\n")
                f.write(f"Max nodes per path: {WalkConfig.MAX_NODES}\n")
                f.write(f"Paths per starting node: {WalkConfig.PATH_COUNT}\n")
                f.write(f"Degree penalty enabled: {WalkConfig.USE_DEGREE_PENALTY}\n")
                f.write(f"Degree penalty beta: {WalkConfig.DEGREE_PENALTY_BETA}\n")
                f.write(f"Probability disturbance: {WalkConfig.USE_DISTURBANCE}\n")
                f.write(f"Disturbance level: {WalkConfig.DISTURBANCE_LEVEL}\n")
                f.write(f"Minimum composite nodes: {WalkConfig.MINIMUM_COMPOSITE_NODES}\n\n")
                
                # Generation Results
                f.write("Generation Results:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total nodes processed: {self.generation_stats['total_nodes']}\n")
                f.write(f"Successful paths: {self.generation_stats['total_paths_generated']}\n")
                f.write(f"Failed attempts: {self.generation_stats['failed_starts']}\n")
                f.write(f"Success rate: {self.generation_stats['success_rate']:.2%}\n")
                f.write(f"Average path length: {self.generation_stats['average_path_length']:.2f}\n")
                f.write(f"Processing time: {self.generation_stats['processing_time']:.2f} seconds\n\n")
                
                # Path Quality Analysis
                if quality_analysis:
                    f.write("Path Quality Analysis:\n")
                    f.write("-" * 30 + "\n")
                    
                    if 'path_statistics' in quality_analysis:
                        stats = quality_analysis['path_statistics']
                        f.write(f"Total paths generated: {stats.get('total_paths', 0)}\n")
                        f.write(f"Path length range: {stats.get('min_length', 0)} - {stats.get('max_length', 0)}\n")
                        f.write(f"Mean path length: {stats.get('mean_length', 0):.2f}\n")
                        f.write(f"Path length std: {stats.get('std_length', 0):.2f}\n\n")
                    
                    if 'node_coverage' in quality_analysis:
                        coverage = quality_analysis['node_coverage']
                        f.write(f"Node coverage rate: {coverage.get('coverage_rate', 0):.2%}\n")
                        f.write(f"Nodes in paths: {coverage.get('nodes_in_paths', 0)}\n")
                        f.write(f"Total graph nodes: {coverage.get('total_graph_nodes', 0)}\n\n")
                    
                    if 'diversity_metrics' in quality_analysis:
                        diversity = quality_analysis['diversity_metrics']
                        f.write(f"Path diversity score: {diversity.get('diversity_score', 0):.3f}\n")
                        f.write(f"Average Jaccard similarity: {diversity.get('average_jaccard_similarity', 0):.3f}\n\n")
                
                # Failure Analysis
                f.write("Failure Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Insufficient neighbors: {self.failure_analysis['insufficient_neighbors']}\n")
                f.write(f"Connectivity issues: {self.failure_analysis['connectivity_issues']}\n")
                f.write(f"Other failures: {self.failure_analysis['other_failures']}\n\n")
                
                if self.failure_analysis['failure_reasons']:
                    f.write("Failure Reasons Summary:\n")
                    for reason, nodes in self.failure_analysis['failure_reasons'].items():
                        f.write(f"  {reason}: {len(nodes)} nodes\n")
            
            logger.info(f"Comprehensive report saved to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return ""
    
    def save_configuration(self) -> str:
        """
        Save the current configuration to JSON file.
        
        Returns:
            str: Path to the saved configuration file
        """
        config_file = self.config_dir / "walk_parameters.json"
        
        config_data = {
            'algorithm': 'Diverse (Degree-Penalized) Random Walk',
            'parameters': {
                'n_hops': WalkConfig.N_HOPS,
                'max_nodes': WalkConfig.MAX_NODES,
                'path_count': WalkConfig.PATH_COUNT,
                'use_degree_penalty': WalkConfig.USE_DEGREE_PENALTY,
                'degree_penalty_beta': WalkConfig.DEGREE_PENALTY_BETA,
                'use_disturbance': WalkConfig.USE_DISTURBANCE,
                'disturbance_level': WalkConfig.DISTURBANCE_LEVEL,
                'minimum_composite_nodes': WalkConfig.MINIMUM_COMPOSITE_NODES,
                'epsilon': WalkConfig.EPSILON
            },
            'performance': {
                'batch_size': WalkConfig.BATCH_SIZE,
                'memory_limit_mb': WalkConfig.MEMORY_LIMIT_MB
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {config_file}")
            return str(config_file)
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return ""
    
    def save_failed_nodes(self) -> str:
        """
        Save information about nodes that failed to generate paths.
        
        Returns:
            str: Path to the saved failed nodes file
        """
        failed_file = self.logs_dir / "skipped_nodes.txt"
        
        try:
            with open(failed_file, 'w', encoding='utf-8') as f:
                f.write("Nodes that failed to generate paths\n")
                f.write("=" * 50 + "\n\n")
                
                for reason, nodes in self.failure_analysis['failure_reasons'].items():
                    f.write(f"Reason: {reason}\n")
                    f.write(f"Count: {len(nodes)}\n")
                    f.write("Nodes:\n")
                    for node in nodes:
                        f.write(f"  - {node}\n")
                    f.write("\n")
            
            logger.info(f"Failed nodes information saved to: {failed_file}")
            return str(failed_file)
            
        except Exception as e:
            logger.error(f"Failed to save failed nodes information: {e}")
            return ""
    
    def generate_diverse_random_walks(self) -> Dict[str, Any]:
        """
        Main method to generate diverse random walks from the knowledge graph.
        
        Returns:
            Dict[str, Any]: Summary of generation results
        """
        logger.info("="*80)
        logger.info("STARTING COGATOM DIVERSE RANDOM WALK GENERATION")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Step 1: Validate input files
            logger.info("\n" + "="*60)
            logger.info("STEP 1: INPUT VALIDATION")
            logger.info("="*60)
            
            if not self.validate_input_files():
                raise FileNotFoundError("Required input files are missing or invalid")
            
            # Step 2: Load knowledge graph
            logger.info("\n" + "="*60)
            logger.info("STEP 2: KNOWLEDGE GRAPH LOADING")
            logger.info("="*60)
            
            if not self.load_knowledge_graph():
                raise ValueError("Failed to load knowledge graph")
            
            if not self.load_graph_statistics():
                logger.warning("Failed to load graph statistics, continuing without them")
            
            # Step 3: Generate random walks
            logger.info("\n" + "="*60)
            logger.info("STEP 3: DIVERSE RANDOM WALK GENERATION")
            logger.info("="*60)
            
            if not self.generate_all_random_walks():
                raise RuntimeError("Failed to generate random walks")
            
            # Step 4: Analyze path quality
            logger.info("\n" + "="*60)
            logger.info("STEP 4: PATH QUALITY ANALYSIS")
            logger.info("="*60)
            
            quality_analysis = self.analyze_path_quality()
            
            # Step 5: Save results
            logger.info("\n" + "="*60)
            logger.info("STEP 5: SAVING RESULTS")
            logger.info("="*60)
            
            paths_file = self.save_paths_to_file()
            stats_file = self.save_statistics(quality_analysis)
            report_file = self.generate_comprehensive_report(quality_analysis)
            config_file = self.save_configuration()
            failed_file = self.save_failed_nodes()
            
            # Final summary
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("\n" + "="*80)
            logger.info("DIVERSE RANDOM WALK GENERATION COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Total processing time: {duration:.2f} seconds")
            logger.info("")
            logger.info("📊 GENERATION SUMMARY:")
            logger.info(f"  Total nodes processed: {self.generation_stats['total_nodes']}")
            logger.info(f"  Successful paths: {self.generation_stats['total_paths_generated']}")
            logger.info(f"  Failed attempts: {self.generation_stats['failed_starts']}")
            logger.info(f"  Success rate: {self.generation_stats['success_rate']:.2%}")
            logger.info(f"  Average path length: {self.generation_stats['average_path_length']:.2f}")
            logger.info("")
            logger.info("📁 OUTPUT FILES:")
            logger.info(f"  Paths: {paths_file}")
            logger.info(f"  Statistics: {stats_file}")
            logger.info(f"  Report: {report_file}")
            logger.info(f"  Configuration: {config_file}")
            logger.info(f"  Failed nodes: {failed_file}")
            logger.info("")
            logger.info("🚀 DIVERSE RANDOM WALKS READY FOR COGNITIVE ATOM COMBINATION!")
            logger.info("="*80)
            
            return {
                'success': True,
                'duration': duration,
                'generation_stats': self.generation_stats,
                'quality_analysis': quality_analysis,
                'output_files': {
                    'paths': paths_file,
                    'statistics': stats_file,
                    'report': report_file,
                    'configuration': config_file,
                    'failed_nodes': failed_file
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Random walk generation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}


def main():
    """
    Main execution function for the CogAtom Diverse Random Walk Generator.
    
    This function initializes the generator and runs the complete pipeline
    for generating diverse random walk paths from the knowledge graph.
    """
    print("CogAtom Diverse Random Walk Generator")
    print("=" * 50)
    print("Generating diverse random walk paths from knowledge graph...")
    print()
    
    # Initialize generator
    generator = CogAtomRandomWalkGenerator()
    
    # Run generation pipeline
    results = generator.generate_diverse_random_walks()
    
    # Print final results
    if results['success']:
        print(f"\n✅ Random walk generation completed successfully in {results['duration']:.2f} seconds")
        print(f"\nGeneration statistics:")
        stats = results['generation_stats']
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Successful paths: {stats['total_paths_generated']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Average path length: {stats['average_path_length']:.2f}")
        print(f"\nOutput directory: {generator.output_dir}")
        return 0
    else:
        print(f"\n❌ Random walk generation failed: {results['error']}")
        return 1


if __name__ == "__main__":
    exit(main())
