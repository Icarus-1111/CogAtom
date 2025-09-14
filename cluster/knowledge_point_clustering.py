#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import time
import argparse
from pathlib import Path
import logging
import warnings
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.metrics.cluster._unsupervised")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root():
    """Get the project root directory (cogatom/)"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    return project_root


class SimilarityBasedClusterer:
    """Similarity-based knowledge point clustering analyzer with 0.9 threshold"""
    
    def __init__(self, config=None):
        """Initialize clusterer with configuration"""
        self.project_root = get_project_root()
        self.config = config or self._get_default_config()
        
        # Data paths
        self.embedding_dir = self.project_root / "data" / "processed" / "embeddings"
        self.output_dir = self.project_root / "data" / "processed" / "clustering"
        
        # Data containers
        self.embeddings = None
        self.knowledge_point_names = []
        self.name_to_index = {}
        self.normalized_embeddings = None
        
        # Results containers
        self.similarity_matrix = None
        self.clustering_results = {}
        
    def _get_default_config(self):
        """Get default clustering configuration"""
        return {
            'similarity_threshold': 0.9,  # Use 0.9 as the primary threshold
            'output': {
                'save_plots': True,
                'save_detailed_results': True
            }
        }
    
    def load_data(self):
        """Load embeddings and knowledge point mappings"""
        logger.info("Loading embedding data and knowledge point mappings...")
        
        # Load embeddings
        embedding_file = self.embedding_dir / "knowledge_point_names_dense.npy"
        if not embedding_file.exists():
            raise FileNotFoundError("Embedding file not found: {}".format(embedding_file))
        
        self.embeddings = np.load(embedding_file, mmap_mode='r')
        logger.info("Loaded embeddings with shape: {}".format(self.embeddings.shape))
        
        # Load knowledge point names mapping
        mapping_file = self.embedding_dir / "knowledge_point_names_list.txt"
        if not mapping_file.exists():
            raise FileNotFoundError("Mapping file not found: {}".format(mapping_file))
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    index = int(parts[0])
                    name = parts[1]
                    self.knowledge_point_names.append(name)
                    self.name_to_index[name] = index
        
        logger.info("Loaded {} knowledge point names".format(len(self.knowledge_point_names)))
        
        # Verify data consistency
        if len(self.knowledge_point_names) != self.embeddings.shape[0]:
            raise ValueError("Mismatch between embeddings ({}) and knowledge points ({})".format(
                self.embeddings.shape[0], len(self.knowledge_point_names)
            ))
        
        # Normalize embeddings
        logger.info("Normalizing embeddings...")
        self.normalized_embeddings = normalize(self.embeddings, norm='l2')
        logger.info("Data loading completed successfully")
    
    def compute_similarity_matrix(self):
        """Compute cosine similarity matrix for all knowledge points"""
        logger.info("Computing cosine similarity matrix...")
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.normalized_embeddings)
        
        logger.info("Similarity matrix computed with shape: {}".format(self.similarity_matrix.shape))
        
        # Print some statistics
        upper_triangle = np.triu(self.similarity_matrix, k=1)
        non_zero_similarities = upper_triangle[upper_triangle > 0]
        
        logger.info("Similarity statistics:")
        logger.info("  Mean similarity: {:.4f}".format(np.mean(non_zero_similarities)))
        logger.info("  Std similarity: {:.4f}".format(np.std(non_zero_similarities)))
        logger.info("  Min similarity: {:.4f}".format(np.min(non_zero_similarities)))
        logger.info("  Max similarity: {:.4f}".format(np.max(non_zero_similarities)))
        
        # Count pairs above different thresholds
        for threshold in [0.95, 0.9, 0.85, 0.8, 0.75]:
            count = np.sum(upper_triangle > threshold)
            percentage = (count / len(non_zero_similarities)) * 100
            logger.info("  Pairs with similarity > {}: {} ({:.2f}%)".format(
                threshold, count, percentage
            ))
    
    def find_connected_components(self, edges):
        """Find connected components in a graph defined by edges"""
        # Build adjacency list
        graph = defaultdict(set)
        nodes = set()
        
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
            nodes.add(u)
            nodes.add(v)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in nodes:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        
        return components
    
    def similarity_based_clustering(self):
        """Perform clustering based on 0.9 similarity threshold"""
        threshold = self.config['similarity_threshold']
        logger.info("Performing similarity-based clustering with threshold {}...".format(threshold))
        
        # Find edges (pairs) above threshold
        edges = []
        n_points = len(self.knowledge_point_names)
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if self.similarity_matrix[i, j] > threshold:
                    edges.append((i, j))
        
        logger.info("Found {} edges above threshold {}".format(len(edges), threshold))
        
        # Find connected components
        if edges:
            clusters = self.find_connected_components(edges)
        else:
            clusters = []  # No connections, no clusters
        
        # Calculate clustered nodes correctly
        clustered_nodes = set()
        for cluster in clusters:
            clustered_nodes.update(cluster)
        
        # Calculate single point count correctly
        single_point_count = n_points - len(clustered_nodes)
        
        # Filter multi-point clusters (all connected components should be multi-point by definition)
        multi_point_clusters = [cluster for cluster in clusters if len(cluster) > 1]
        
        logger.info("Clustering results for threshold {}:".format(threshold))
        logger.info("  Total knowledge points: {}".format(n_points))
        logger.info("  Connected components found: {}".format(len(clusters)))
        logger.info("  Multi-point clusters: {}".format(len(multi_point_clusters)))
        logger.info("  Points in clusters: {}".format(len(clustered_nodes)))
        logger.info("  Single points (unclustered): {}".format(single_point_count))
        
        # Calculate cluster size statistics
        if multi_point_clusters:
            cluster_sizes = [len(cluster) for cluster in multi_point_clusters]
            logger.info("  Cluster size statistics:")
            logger.info("    Mean size: {:.2f}".format(np.mean(cluster_sizes)))
            logger.info("    Max size: {}".format(np.max(cluster_sizes)))
            logger.info("    Min size: {}".format(np.min(cluster_sizes)))
            
            # Show largest clusters
            sorted_clusters = sorted(multi_point_clusters, key=len, reverse=True)
            logger.info("  Top 5 largest clusters:")
            for i, cluster in enumerate(sorted_clusters[:5]):
                main_name = self.knowledge_point_names[cluster[0]]
                logger.info("    Cluster {}: {} points, main: '{}'".format(
                    i+1, len(cluster), main_name[:50] + "..." if len(main_name) > 50 else main_name
                ))
        
        # Store results with correct calculations
        self.clustering_results = {
            'clusters': multi_point_clusters,
            'num_clusters': len(multi_point_clusters),
            'total_points_in_clusters': len(clustered_nodes),
            'single_point_count': single_point_count,
            'compression_ratio': n_points / len(multi_point_clusters) if multi_point_clusters else float('inf')
        }
        
        return multi_point_clusters
    
    def generate_cluster_results(self):
        """Generate clustering results in the required format"""
        threshold = self.config['similarity_threshold']
        logger.info("Generating cluster results for threshold {}...".format(threshold))
        
        clusters = self.clustering_results['clusters']
        results = []
        
        for cluster_id, member_indices in enumerate(tqdm(clusters, desc="Processing clusters")):
            if len(member_indices) < 2:
                continue  # Skip single-member clusters (shouldn't happen here)
            
            # Use the first point as the main representative
            main_index = member_indices[0]
            main_keyword = self.knowledge_point_names[main_index]
            
            # Extract descriptions (assuming format "name - description")
            descriptions = []
            contents = []
            for idx in member_indices:
                kp_name = self.knowledge_point_names[idx]
                contents.append(kp_name)
                if ' - ' in kp_name:
                    description = kp_name.split(' - ', 1)[1]
                    descriptions.append(description)
                else:
                    descriptions.append("")
            
            main_description = descriptions[0] if descriptions[0] else ""
            
            # Create result entry in the required format
            result_entry = {
                'main_keyword': main_keyword,
                'main_index': main_index,
                'main_description': main_description,
                'indices': member_indices,
                'descriptions': descriptions,
                'topic_indices': member_indices,  # Same as indices for compatibility
                'contents': contents
            }
            
            results.append(result_entry)
        
        logger.info("Generated {} cluster groups for threshold {}".format(len(results), threshold))
        return results
    
    def save_results(self, cluster_results):
        """Save clustering results and analysis"""
        threshold = self.config['similarity_threshold']
        logger.info("Saving clustering results for threshold {}...".format(threshold))
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir = self.output_dir / "similarity_analysis"
        results_dir = self.output_dir / "similarity_results"
        analysis_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        
        # Save cluster results file in the required format (compatible with cooccurrence analysis)
        data_name = "atom2olympiad"
        output_filename = "{}_output_clusters_{}_threshold_{}.txt".format(
            data_name, self.clustering_results['num_clusters'], int(threshold * 100)
        )
        
        output_file = results_dir / output_filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in cluster_results:
                line = "[main_keyword: \"{}\"]##[main_index: {}]##[main_description: \"{}\"]##[indices: {}]##[descriptions: {}]##[topics: {}]##[contents: {}]\n".format(
                    entry['main_keyword'],
                    entry['main_index'],
                    entry['main_description'],
                    entry['indices'],
                    entry['descriptions'],
                    entry['topic_indices'],
                    entry['contents']
                )
                f.write(line)
        
        logger.info("Saved similarity clustering results to: {}".format(output_file))
        
        # Save detailed analysis with correct calculations
        analysis_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_knowledge_points': len(self.knowledge_point_names),
            'similarity_threshold': threshold,
            'clustering_results': {
                'num_clusters': self.clustering_results['num_clusters'],
                'total_points_in_clusters': self.clustering_results['total_points_in_clusters'],
                'single_point_count': self.clustering_results['single_point_count'],
                'compression_ratio': self.clustering_results['compression_ratio'],
                'percentage_clustered': (self.clustering_results['total_points_in_clusters'] / len(self.knowledge_point_names)) * 100,
                'final_knowledge_points': self.clustering_results['single_point_count'] + self.clustering_results['num_clusters']
            },
            'output_file': str(output_file)
        }
        
        # Save analysis file
        analysis_file = analysis_dir / "similarity_clustering_analysis_09.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Saved analysis data to: {}".format(analysis_file))
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print a summary of clustering results with correct calculations"""
        threshold = self.config['similarity_threshold']
        result = self.clustering_results
        
        logger.info("\n" + "="*60)
        logger.info("SIMILARITY-BASED CLUSTERING SUMMARY (Threshold 0.9)")
        logger.info("="*60)
        logger.info("Total original knowledge points: {}".format(len(self.knowledge_point_names)))
        logger.info("Similarity threshold: {}".format(threshold))
        logger.info("")
        logger.info("Clustering Results:")
        logger.info("  Number of clusters: {}".format(result['num_clusters']))
        logger.info("  Points in clusters: {}".format(result['total_points_in_clusters']))
        logger.info("  Single points (unclustered): {}".format(result['single_point_count']))
        logger.info("  Percentage clustered: {:.1f}%".format(
            (result['total_points_in_clusters'] / len(self.knowledge_point_names)) * 100
        ))
        logger.info("")
        logger.info("Final Knowledge Points After Clustering:")
        final_points = result['single_point_count'] + result['num_clusters']
        compression_rate = (1 - final_points / len(self.knowledge_point_names)) * 100
        logger.info("  Cluster representatives: {}".format(result['num_clusters']))
        logger.info("  Unclustered individual points: {}".format(result['single_point_count']))
        logger.info("  Total final knowledge points: {} (was {})".format(final_points, len(self.knowledge_point_names)))
        logger.info("  Overall compression: {:.1f}% reduction".format(compression_rate))
        logger.info("")
        logger.info("Interpretation:")
        logger.info("  - {} clusters represent groups of similar knowledge points".format(result['num_clusters']))
        logger.info("  - {} individual points have no highly similar neighbors".format(result['single_point_count']))
        logger.info("  - Final dataset contains {} unique knowledge concepts".format(final_points))
        logger.info("="*60)
    
    def run_similarity_clustering(self):
        """Run the complete similarity-based clustering pipeline"""
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Compute similarity matrix
            self.compute_similarity_matrix()
            
            # Step 3: Perform clustering
            self.similarity_based_clustering()
            
            # Step 4: Generate results
            cluster_results = self.generate_cluster_results()
            
            # Step 5: Save results
            self.save_results(cluster_results)
            
            total_time = time.time() - start_time
            logger.info("Similarity-based clustering completed successfully in {:.2f} seconds".format(total_time))
            
            return True
            
        except Exception as e:
            logger.error("Similarity-based clustering failed: {}".format(e))
            return False


def main():
    """Main function for similarity-based clustering analysis"""
    parser = argparse.ArgumentParser(description='Similarity-Based Knowledge Point Clustering (0.9 Threshold)')
    parser.add_argument('--threshold', type=float, default=0.9,
                       help='Similarity threshold (default: 0.9)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = {
        'similarity_threshold': args.threshold,
        'output': {
            'save_plots': not args.no_plots,
            'save_detailed_results': True
        }
    }
    
    # Run clustering
    clusterer = SimilarityBasedClusterer(config)
    success = clusterer.run_similarity_clustering()
    
    if success:
        print("\n✅ Similarity-based clustering (threshold 0.9) completed successfully!")
        print("Results saved to: data/processed/clustering/similarity_results/")
        print("Analysis saved to: data/processed/clustering/similarity_analysis/")
        print("\nNext step: Use the generated cluster file for cooccurrence analysis")
        print("Parameters for cooccurrence analysis:")
        print("  - n_clusters: {} (from the output filename)".format(clusterer.clustering_results['num_clusters']))
        print("  - similarity_threshold: 0.9")
        print("  - Final knowledge points: {} (was 3169)".format(
            clusterer.clustering_results['single_point_count'] + clusterer.clustering_results['num_clusters']
        ))
    else:
        print("\n❌ Similarity-based clustering failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
