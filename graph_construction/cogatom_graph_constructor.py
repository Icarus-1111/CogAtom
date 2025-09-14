#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx
import numpy as np
from tqdm import tqdm

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('graph_construction.log')
    ]
)
logger = logging.getLogger(__name__)


class GraphConfig:
    """Configuration class for graph construction parameters."""
    
    # File paths (relative to project structure)
    COOCCURRENCE_FILE = "aggregated_cooccurrence_247_90.txt"
    CLUSTER_MAPPING_FILE = "cluster_mapping_247_90.json"
    
    # Algorithm parameters
    LAMBDA_PARAM = 2.0  # Statistical threshold multiplier for supernode detection
    LOG_TRANSFORM = True  # Apply log1p transformation to weights
    MIN_EDGE_WEIGHT = 1  # Minimum edge weight to include
    
    # Output configuration
    EXPORT_GEXF = True  # Export GEXF format for Gephi
    EXPORT_EDGELIST = True  # Export EdgeList format
    EXPORT_ANALYSIS = True  # Generate analysis reports
    
    # Graph analysis parameters
    ANALYZE_CONNECTIVITY = True
    ANALYZE_CENTRALITY = True
    ANALYZE_COMMUNITIES = False  # Computationally expensive for large graphs


class CogAtomGraphConstructor:
    """
    Main class for constructing knowledge graphs from cooccurrence data.
    
    This class handles the complete pipeline from data loading to graph export,
    including weight transformation, duplicate edge detection, supernode pruning,
    and comprehensive analysis.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the graph constructor.
        
        Args:
            project_root: Optional path to project root. If None, auto-detect from script location.
        """
        # Setup paths using relative path resolution
        self.script_dir = Path(__file__).parent
        self.project_root = Path(project_root) if project_root else self.script_dir.parent
        self.data_dir = self.project_root / "data"
        
        # Input paths
        self.cooccurrence_dir = self.data_dir / "processed" / "graphs" / "cooccurrence"
        self.aggregated_file = (
            self.cooccurrence_dir / "aggregated_cooccurrence" / "cogatom" / 
            GraphConfig.COOCCURRENCE_FILE
        )
        self.cluster_mapping_file = (
            self.cooccurrence_dir / "analysis_results" / "cogatom" / 
            GraphConfig.CLUSTER_MAPPING_FILE
        )
        
        # Output paths
        self.output_base = self.data_dir / "processed" / "knowledge_graphs"
        self.output_dir = self.output_base / "cogatom_247_90"
        self.original_dir = self.output_dir / "original"
        self.cleaned_dir = self.output_dir / "cleaned"
        self.analysis_dir = self.output_dir / "analysis"
        
        # Create output directories
        for dir_path in [self.original_dir, self.cleaned_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.edges: List[Tuple[str, str]] = []
        self.weights: List[float] = []
        self.transformed_weights: List[float] = []
        self.cluster_mapping: Dict[str, Any] = {}
        self.original_graph: Optional[nx.Graph] = None
        self.cleaned_graph: Optional[nx.Graph] = None
        
        # Analysis results
        self.duplicate_edges_info: Dict[str, Any] = {}
        self.supernode_info: Dict[str, Any] = {}
        self.graph_statistics: Dict[str, Any] = {}
        
        logger.info(f"CogAtom Graph Constructor initialized")
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
            ("Aggregated cooccurrence data", self.aggregated_file),
            ("Cluster mapping data", self.cluster_mapping_file)
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
    
    def load_cooccurrence_data(self) -> bool:
        """
        Load cooccurrence data from the aggregated file.
        
        Expected format: "node1##node2 - weight"
        
        Returns:
            bool: True if loading successful, False otherwise.
        """
        logger.info(f"Loading cooccurrence data from: {self.aggregated_file}")
        
        try:
            self.edges = []
            self.weights = []
            weight_distribution = Counter()
            
            with open(self.aggregated_file, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(tqdm(file, desc="Loading cooccurrence data"), 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse format: "node1##node2 - weight"
                        parts = line.split(' - ')
                        if len(parts) != 2:
                            logger.warning(f"Invalid format at line {line_num}: {line}")
                            continue
                        
                        node_pair = parts[0].split('##')
                        if len(node_pair) != 2:
                            logger.warning(f"Invalid node pair at line {line_num}: {parts[0]}")
                            continue
                        
                        weight = float(parts[1])
                        if weight < GraphConfig.MIN_EDGE_WEIGHT:
                            continue  # Skip edges below minimum weight
                        
                        # Ensure consistent edge representation (sorted)
                        edge = tuple(sorted(node_pair))
                        self.edges.append(edge)
                        self.weights.append(weight)
                        weight_distribution[int(weight)] += 1
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(self.edges)} edges with weights")
            logger.info("Weight distribution:")
            for weight, count in sorted(weight_distribution.items())[:10]:
                logger.info(f"  Weight {weight}: {count} edges")
            
            if len(weight_distribution) > 10:
                logger.info(f"  ... and {len(weight_distribution) - 10} more weight levels")
            
            return len(self.edges) > 0
            
        except Exception as e:
            logger.error(f"Failed to load cooccurrence data: {e}")
            return False
    
    def load_cluster_mapping(self) -> bool:
        """
        Load cluster mapping information for node attributes.
        
        Returns:
            bool: True if loading successful, False otherwise.
        """
        logger.info(f"Loading cluster mapping from: {self.cluster_mapping_file}")
        
        try:
            with open(self.cluster_mapping_file, 'r', encoding='utf-8') as file:
                self.cluster_mapping = json.load(file)
            
            logger.info(f"Loaded cluster mapping with {len(self.cluster_mapping.get('original_to_representative', {}))} entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cluster mapping: {e}")
            return False
    
    def transform_weights(self) -> None:
        """
        Apply log1p transformation to edge weights to reduce skewness.
        
        The log1p transformation helps handle the long-tail distribution
        commonly found in cooccurrence data.
        """
        logger.info("Applying log1p transformation to edge weights...")
        
        if not GraphConfig.LOG_TRANSFORM:
            self.transformed_weights = self.weights.copy()
            logger.info("Log transformation disabled, using original weights")
            return
        
        original_weights = np.array(self.weights)
        self.transformed_weights = np.log1p(original_weights).tolist()
        
        logger.info(f"Weight transformation completed:")
        logger.info(f"  Original range: [{np.min(original_weights):.3f}, {np.max(original_weights):.3f}]")
        logger.info(f"  Transformed range: [{np.min(self.transformed_weights):.3f}, {np.max(self.transformed_weights):.3f}]")
    
    def detect_duplicate_edges(self) -> Dict[str, Any]:
        """
        Detect and analyze duplicate edges in the dataset.
        
        Returns:
            dict: Information about duplicate edges found.
        """
        logger.info("Detecting duplicate edges...")
        
        edge_records = defaultdict(list)
        
        # Group edges by their identity
        for idx, (edge, weight) in enumerate(zip(self.edges, self.transformed_weights)):
            edge_records[edge].append((idx, weight))
        
        # Analyze duplicates
        duplicate_edges = []
        total_duplicates = 0
        
        for edge, records in edge_records.items():
            if len(records) > 1:
                duplicate_edges.append({
                    'edge': edge,
                    'occurrences': len(records),
                    'positions': [pos for pos, _ in records],
                    'weights': [weight for _, weight in records],
                    'total_weight': sum(weight for _, weight in records)
                })
                total_duplicates += len(records) - 1
        
        duplicate_info = {
            'total_duplicate_edges': len(duplicate_edges),
            'total_duplicate_occurrences': total_duplicates,
            'duplicate_details': duplicate_edges
        }
        
        # Save duplicate edges report
        duplicate_report_file = self.original_dir / "duplicate_edges_report.txt"
        with open(duplicate_report_file, 'w', encoding='utf-8') as f:
            f.write("Duplicate Edges Detection Report\n")
            f.write("=" * 50 + "\n\n")
            
            if duplicate_edges:
                f.write(f"Found {len(duplicate_edges)} duplicate edges with {total_duplicates} total duplicates\n\n")
                
                for dup in duplicate_edges:
                    f.write(f"Edge: {dup['edge'][0]} -- {dup['edge'][1]}\n")
                    f.write(f"  Occurrences: {dup['occurrences']}\n")
                    f.write(f"  Positions: {dup['positions']}\n")
                    f.write(f"  Individual weights: {dup['weights']}\n")
                    f.write(f"  Combined weight: {dup['total_weight']:.4f}\n\n")
            else:
                f.write("No duplicate edges found.\n")
        
        logger.info(f"Duplicate edge analysis completed:")
        logger.info(f"  Duplicate edges: {len(duplicate_edges)}")
        logger.info(f"  Total duplicates: {total_duplicates}")
        logger.info(f"  Report saved to: {duplicate_report_file}")
        
        self.duplicate_edges_info = duplicate_info
        return duplicate_info
    
    def build_original_graph(self) -> nx.Graph:
        """
        Build the original weighted graph from cooccurrence data.
        
        Returns:
            nx.Graph: The constructed graph with all edges and attributes.
        """
        logger.info("Building original weighted graph...")
        
        G = nx.Graph()
        edge_weights = defaultdict(float)
        edge_counts = defaultdict(int)
        
        # Accumulate weights for duplicate edges
        for edge, weight in zip(self.edges, self.transformed_weights):
            edge_weights[edge] += weight
            edge_counts[edge] += 1
        
        # Add edges to graph
        for edge, total_weight in edge_weights.items():
            G.add_edge(
                edge[0], edge[1],
                weight=total_weight,
                original_occurrences=edge_counts[edge],
                edge_type='cooccurrence'
            )
        
        # Add node attributes from cluster mapping
        if self.cluster_mapping:
            original_to_rep = self.cluster_mapping.get('original_to_representative', {})
            rep_to_members = self.cluster_mapping.get('representative_to_members', {})
            
            for node in G.nodes():
                cluster_rep = original_to_rep.get(node, node)
                is_representative = node in rep_to_members
                cluster_size = len(rep_to_members.get(node, [node]))
                
                G.nodes[node].update({
                    'cluster_representative': cluster_rep,
                    'is_cluster_representative': is_representative,
                    'cluster_size': cluster_size,
                    'node_type': 'knowledge_point'
                })
        
        logger.info(f"Original graph constructed:")
        logger.info(f"  Nodes: {G.number_of_nodes()}")
        logger.info(f"  Edges: {G.number_of_edges()}")
        logger.info(f"  Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
        
        self.original_graph = G
        return G
    
    def identify_supernodes(self, graph: nx.Graph) -> Tuple[List[str], Dict[str, float]]:
        """
        Identify supernodes based on statistical degree analysis.
        
        Args:
            graph: The graph to analyze.
            
        Returns:
            tuple: (list of supernode names, statistics dict)
        """
        logger.info("Identifying supernodes using statistical thresholds...")
        
        # Calculate degree statistics
        degrees = np.array([degree for node, degree in graph.degree()])
        mean_degree = degrees.mean()
        std_degree = degrees.std()
        threshold = mean_degree + GraphConfig.LAMBDA_PARAM * std_degree
        
        # Identify supernodes
        supernodes = [
            node for node, degree in graph.degree() 
            if degree > threshold
        ]
        
        # Collect statistics
        stats = {
            'total_nodes': graph.number_of_nodes(),
            'mean_degree': float(mean_degree),
            'std_degree': float(std_degree),
            'threshold': float(threshold),
            'lambda_param': GraphConfig.LAMBDA_PARAM,
            'supernodes_count': len(supernodes),
            'supernodes_percentage': (len(supernodes) / graph.number_of_nodes()) * 100,
            'supernode_degrees': {node: graph.degree(node) for node in supernodes}
        }
        
        logger.info(f"Supernode identification completed:")
        logger.info(f"  Mean degree: {mean_degree:.2f}")
        logger.info(f"  Std deviation: {std_degree:.2f}")
        logger.info(f"  Threshold (μ + {GraphConfig.LAMBDA_PARAM}σ): {threshold:.2f}")
        logger.info(f"  Supernodes found: {len(supernodes)} ({stats['supernodes_percentage']:.1f}%)")
        
        if supernodes:
            logger.info("Top supernodes by degree:")
            sorted_supernodes = sorted(supernodes, key=lambda x: graph.degree(x), reverse=True)
            for node in sorted_supernodes[:5]:
                logger.info(f"  {node}: degree {graph.degree(node)}")
        
        self.supernode_info = stats
        return supernodes, stats
    
    def build_cleaned_graph(self, original_graph: nx.Graph) -> nx.Graph:
        """
        Build cleaned graph by removing supernodes.
        
        Args:
            original_graph: The original graph to clean.
            
        Returns:
            nx.Graph: The cleaned graph with supernodes removed.
        """
        logger.info("Building cleaned graph by removing supernodes...")
        
        # Identify supernodes
        supernodes, supernode_stats = self.identify_supernodes(original_graph)
        
        # Create cleaned graph
        cleaned_graph = original_graph.copy()
        cleaned_graph.remove_nodes_from(supernodes)
        
        logger.info(f"Cleaned graph created:")
        logger.info(f"  Original: {original_graph.number_of_nodes()} nodes, {original_graph.number_of_edges()} edges")
        logger.info(f"  Cleaned: {cleaned_graph.number_of_nodes()} nodes, {cleaned_graph.number_of_edges()} edges")
        logger.info(f"  Removed: {len(supernodes)} supernodes")
        
        # Save supernode analysis
        supernode_file = self.analysis_dir / "supernode_analysis.txt"
        with open(supernode_file, 'w', encoding='utf-8') as f:
            f.write("Supernode Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Statistical Parameters:\n")
            f.write(f"  Mean degree: {supernode_stats['mean_degree']:.2f}\n")
            f.write(f"  Standard deviation: {supernode_stats['std_degree']:.2f}\n")
            f.write(f"  Lambda parameter: {supernode_stats['lambda_param']}\n")
            f.write(f"  Threshold: {supernode_stats['threshold']:.2f}\n\n")
            
            f.write(f"Results:\n")
            f.write(f"  Supernodes identified: {len(supernodes)}\n")
            f.write(f"  Percentage of total nodes: {supernode_stats['supernodes_percentage']:.1f}%\n\n")
            
            if supernodes:
                f.write("Supernodes (sorted by degree):\n")
                sorted_supernodes = sorted(supernodes, key=lambda x: original_graph.degree(x), reverse=True)
                for node in sorted_supernodes:
                    f.write(f"  {node}: degree {original_graph.degree(node)}\n")
        
        logger.info(f"Supernode analysis saved to: {supernode_file}")
        
        self.cleaned_graph = cleaned_graph
        return cleaned_graph
    
    def analyze_graph_properties(self, graph: nx.Graph, graph_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive graph analysis.
        
        Args:
            graph: The graph to analyze.
            graph_name: Name identifier for the graph.
            
        Returns:
            dict: Comprehensive analysis results.
        """
        logger.info(f"Analyzing properties of {graph_name} graph...")
        
        analysis = {
            'basic_properties': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'is_connected': nx.is_connected(graph)
            }
        }
        
        # Degree analysis
        degrees = [degree for node, degree in graph.degree()]
        analysis['degree_statistics'] = {
            'mean': float(np.mean(degrees)),
            'std': float(np.std(degrees)),
            'min': int(np.min(degrees)),
            'max': int(np.max(degrees)),
            'median': float(np.median(degrees))
        }
        
        # Weight analysis
        if graph.edges():
            weights = [data.get('weight', 1.0) for _, _, data in graph.edges(data=True)]
            analysis['weight_statistics'] = {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'median': float(np.median(weights))
            }
        
        # Connectivity analysis
        if GraphConfig.ANALYZE_CONNECTIVITY:
            if nx.is_connected(graph):
                analysis['connectivity'] = {
                    'diameter': nx.diameter(graph),
                    'average_shortest_path_length': nx.average_shortest_path_length(graph),
                    'radius': nx.radius(graph)
                }
            else:
                components = list(nx.connected_components(graph))
                analysis['connectivity'] = {
                    'connected_components': len(components),
                    'largest_component_size': len(max(components, key=len)),
                    'component_sizes': [len(comp) for comp in components]
                }
        
        # Centrality analysis (for smaller graphs)
        if GraphConfig.ANALYZE_CENTRALITY and graph.number_of_nodes() < 5000:
            logger.info("Computing centrality measures...")
            
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph, k=min(1000, graph.number_of_nodes()))
            
            # Get top nodes by centrality
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            analysis['centrality'] = {
                'top_degree_centrality': [(node, float(score)) for node, score in top_degree],
                'top_betweenness_centrality': [(node, float(score)) for node, score in top_betweenness]
            }
        
        logger.info(f"{graph_name} graph analysis completed")
        return analysis
    
    def export_graph_formats(self) -> Dict[str, str]:
        """
        Export graphs in multiple formats.
        
        Returns:
            dict: Mapping of format names to file paths.
        """
        logger.info("Exporting graphs in multiple formats...")
        
        exported_files = {}
        
        # Export original graph
        if self.original_graph and GraphConfig.EXPORT_GEXF:
            original_gexf = self.original_dir / "knowledge_graph_original.gexf"
            nx.write_gexf(self.original_graph, original_gexf)
            exported_files['original_gexf'] = str(original_gexf)
            logger.info(f"Original graph exported to GEXF: {original_gexf}")
        
        # Export cleaned graph
        if self.cleaned_graph:
            if GraphConfig.EXPORT_GEXF:
                cleaned_gexf = self.cleaned_dir / "knowledge_graph_cleaned.gexf"
                nx.write_gexf(self.cleaned_graph, cleaned_gexf)
                exported_files['cleaned_gexf'] = str(cleaned_gexf)
                logger.info(f"Cleaned graph exported to GEXF: {cleaned_gexf}")
            
            if GraphConfig.EXPORT_EDGELIST:
                cleaned_edgelist = self.cleaned_dir / "knowledge_graph_cleaned.edgelist"
                nx.write_weighted_edgelist(self.cleaned_graph, cleaned_edgelist)
                exported_files['cleaned_edgelist'] = str(cleaned_edgelist)
                logger.info(f"Cleaned graph exported to EdgeList: {cleaned_edgelist}")
        
        return exported_files
    
    def generate_comprehensive_report(self, exported_files: Dict[str, str]) -> str:
        """
        Generate a comprehensive construction report.
        
        Args:
            exported_files: Dictionary of exported file paths.
            
        Returns:
            str: Path to the generated report file.
        """
        logger.info("Generating comprehensive construction report...")
        
        report_file = self.analysis_dir / "construction_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CogAtom Knowledge Graph Construction Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Construction Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Project: CogAtom\n")
            f.write(f"Configuration: 247 clusters, 90% similarity threshold\n\n")
            
            # Input data summary
            f.write("Input Data Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Cooccurrence file: {self.aggregated_file.name}\n")
            f.write(f"Cluster mapping file: {self.cluster_mapping_file.name}\n")
            f.write(f"Total edges loaded: {len(self.edges)}\n")
            f.write(f"Weight transformation: {'log1p' if GraphConfig.LOG_TRANSFORM else 'none'}\n\n")
            
            # Duplicate edges summary
            if self.duplicate_edges_info:
                f.write("Duplicate Edges Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Duplicate edges found: {self.duplicate_edges_info['total_duplicate_edges']}\n")
                f.write(f"Total duplicates: {self.duplicate_edges_info['total_duplicate_occurrences']}\n\n")
            
            # Graph statistics
            if self.original_graph:
                f.write("Original Graph:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Nodes: {self.original_graph.number_of_nodes()}\n")
                f.write(f"Edges: {self.original_graph.number_of_edges()}\n")
                f.write(f"Density: {nx.density(self.original_graph):.6f}\n")
                f.write(f"Connected: {nx.is_connected(self.original_graph)}\n\n")
            
            if self.cleaned_graph:
                f.write("Cleaned Graph (after supernode removal):\n")
                f.write("-" * 30 + "\n")
                f.write(f"Nodes: {self.cleaned_graph.number_of_nodes()}\n")
                f.write(f"Edges: {self.cleaned_graph.number_of_edges()}\n")
                f.write(f"Density: {nx.density(self.cleaned_graph):.6f}\n")
                f.write(f"Connected: {nx.is_connected(self.cleaned_graph)}\n\n")
            
            # Supernode analysis
            if self.supernode_info:
                f.write("Supernode Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Detection threshold: {self.supernode_info['threshold']:.2f}\n")
                f.write(f"Supernodes removed: {self.supernode_info['supernodes_count']}\n")
                f.write(f"Percentage removed: {self.supernode_info['supernodes_percentage']:.1f}%\n\n")
            
            # Exported files
            f.write("Exported Files:\n")
            f.write("-" * 30 + "\n")
            for format_name, file_path in exported_files.items():
                f.write(f"{format_name}: {Path(file_path).name}\n")
            f.write("\n")
            
            # Configuration used
            f.write("Configuration Parameters:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Lambda parameter: {GraphConfig.LAMBDA_PARAM}\n")
            f.write(f"Minimum edge weight: {GraphConfig.MIN_EDGE_WEIGHT}\n")
            f.write(f"Log transformation: {GraphConfig.LOG_TRANSFORM}\n")
            f.write(f"Export GEXF: {GraphConfig.EXPORT_GEXF}\n")
            f.write(f"Export EdgeList: {GraphConfig.EXPORT_EDGELIST}\n")
        
        logger.info(f"Comprehensive report saved to: {report_file}")
        return str(report_file)
    
    def save_graph_statistics(self) -> str:
        """
        Save detailed graph statistics in JSON format.
        
        Returns:
            str: Path to the statistics file.
        """
        logger.info("Saving detailed graph statistics...")
        
        statistics = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'project': 'CogAtom',
                'configuration': 'cogatom_247_90',
                'constructor_version': '1.0.0'
            },
            'input_data': {
                'total_edges_loaded': len(self.edges),
                'weight_transformation': 'log1p' if GraphConfig.LOG_TRANSFORM else 'none',
                'min_edge_weight_threshold': GraphConfig.MIN_EDGE_WEIGHT
            },
            'duplicate_edges': self.duplicate_edges_info,
            'supernode_analysis': self.supernode_info
        }
        
        # Add graph analyses
        if self.original_graph:
            statistics['original_graph'] = self.analyze_graph_properties(
                self.original_graph, 'original'
            )
        
        if self.cleaned_graph:
            statistics['cleaned_graph'] = self.analyze_graph_properties(
                self.cleaned_graph, 'cleaned'
            )
        
        # Save statistics
        stats_file = self.analysis_dir / "graph_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Graph statistics saved to: {stats_file}")
        self.graph_statistics = statistics
        return str(stats_file)
    
    def construct_knowledge_graphs(self) -> Dict[str, Any]:
        """
        Main method to construct knowledge graphs from cooccurrence data.
        
        Returns:
            dict: Summary of construction results.
        """
        logger.info("="*80)
        logger.info("STARTING COGATOM KNOWLEDGE GRAPH CONSTRUCTION")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Step 1: Validate input files
            logger.info("\n" + "="*60)
            logger.info("STEP 1: INPUT VALIDATION")
            logger.info("="*60)
            
            if not self.validate_input_files():
                raise FileNotFoundError("Required input files are missing or invalid")
            
            # Step 2: Load data
            logger.info("\n" + "="*60)
            logger.info("STEP 2: DATA LOADING")
            logger.info("="*60)
            
            if not self.load_cooccurrence_data():
                raise ValueError("Failed to load cooccurrence data")
            
            if not self.load_cluster_mapping():
                logger.warning("Failed to load cluster mapping, continuing without node attributes")
            
            # Step 3: Transform weights
            logger.info("\n" + "="*60)
            logger.info("STEP 3: WEIGHT TRANSFORMATION")
            logger.info("="*60)
            
            self.transform_weights()
            
            # Step 4: Detect duplicate edges
            logger.info("\n" + "="*60)
            logger.info("STEP 4: DUPLICATE EDGE DETECTION")
            logger.info("="*60)
            
            self.detect_duplicate_edges()
            
            # Step 5: Build original graph
            logger.info("\n" + "="*60)
            logger.info("STEP 5: ORIGINAL GRAPH CONSTRUCTION")
            logger.info("="*60)
            
            self.build_original_graph()
            
            # Step 6: Build cleaned graph
            logger.info("\n" + "="*60)
            logger.info("STEP 6: CLEANED GRAPH CONSTRUCTION")
            logger.info("="*60)
            
            self.build_cleaned_graph(self.original_graph)
            
            # Step 7: Export graphs
            logger.info("\n" + "="*60)
            logger.info("STEP 7: GRAPH EXPORT")
            logger.info("="*60)
            
            exported_files = self.export_graph_formats()
            
            # Step 8: Generate reports
            logger.info("\n" + "="*60)
            logger.info("STEP 8: ANALYSIS AND REPORTING")
            logger.info("="*60)
            
            stats_file = self.save_graph_statistics()
            report_file = self.generate_comprehensive_report(exported_files)
            
            # Final summary
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("\n" + "="*80)
            logger.info("KNOWLEDGE GRAPH CONSTRUCTION COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Total processing time: {duration:.2f} seconds")
            logger.info("")
            logger.info("📊 CONSTRUCTION SUMMARY:")
            logger.info(f"  Input edges processed: {len(self.edges)}")
            logger.info(f"  Original graph: {self.original_graph.number_of_nodes()} nodes, {self.original_graph.number_of_edges()} edges")
            logger.info(f"  Cleaned graph: {self.cleaned_graph.number_of_nodes()} nodes, {self.cleaned_graph.number_of_edges()} edges")
            logger.info(f"  Supernodes removed: {self.supernode_info.get('supernodes_count', 0)}")
            logger.info("")
            logger.info("📁 OUTPUT FILES:")
            for format_name, file_path in exported_files.items():
                logger.info(f"  {format_name}: {file_path}")
            logger.info(f"  Statistics: {stats_file}")
            logger.info(f"  Report: {report_file}")
            logger.info("")
            logger.info("🚀 GRAPHS READY FOR ANALYSIS AND VISUALIZATION!")
            logger.info("="*80)
            
            return {
                'success': True,
                'duration': duration,
                'original_graph_stats': {
                    'nodes': self.original_graph.number_of_nodes(),
                    'edges': self.original_graph.number_of_edges()
                },
                'cleaned_graph_stats': {
                    'nodes': self.cleaned_graph.number_of_nodes(),
                    'edges': self.cleaned_graph.number_of_edges()
                },
                'exported_files': exported_files,
                'analysis_files': {
                    'statistics': stats_file,
                    'report': report_file
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Graph construction failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}


def main():
    """
    Main execution function for the CogAtom Knowledge Graph Constructor.
    
    This function initializes the constructor and runs the complete pipeline
    for building knowledge graphs from cooccurrence data.
    """
    print("CogAtom Knowledge Graph Constructor")
    print("=" * 50)
    print("Building knowledge graphs from cooccurrence data...")
    print()
    
    # Initialize constructor
    constructor = CogAtomGraphConstructor()
    
    # Run construction pipeline
    results = constructor.construct_knowledge_graphs()
    
    # Print final results
    if results['success']:
        print(f"\n✅ Graph construction completed successfully in {results['duration']:.2f} seconds")
        print("\nGenerated graphs:")
        print(f"  Original: {results['original_graph_stats']['nodes']} nodes, {results['original_graph_stats']['edges']} edges")
        print(f"  Cleaned: {results['cleaned_graph_stats']['nodes']} nodes, {results['cleaned_graph_stats']['edges']} edges")
        print(f"\nOutput directory: {constructor.output_dir}")
        return 0
    else:
        print(f"\n❌ Graph construction failed: {results['error']}")
        return 1


if __name__ == "__main__":
    exit(main())
