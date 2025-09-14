#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import pickle
import ast
import os
import time
import logging
from pathlib import Path
from itertools import combinations
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CogAtomCooccurrenceAnalyzer:
    """Complete cooccurrence analysis for CogAtom knowledge points with clustering aggregation"""
    
    def __init__(self, project_root=None):
        """Initialize analyzer with project paths"""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.data_dir = self.project_root / "data"
        
        # Input paths
        self.knowledge_list_file = self.data_dir / "processed" / "knowledge_points" / "processed" / "cogatom_knowledge_list.json"
        self.cluster_file = self.data_dir / "processed" / "clustering" / "similarity_results" / "atom2olympiad_output_clusters_247_threshold_90.txt"
        
        # Output paths
        self.output_base = self.data_dir / "processed" / "graphs" / "cooccurrence"
        self.initial_output_dir = self.output_base / "initial_cooccurrence" / "cogatom"
        self.aggregated_output_dir = self.output_base / "aggregated_cooccurrence" / "cogatom"
        self.analysis_output_dir = self.output_base / "analysis_results" / "cogatom"
        
        # Create output directories
        for dir_path in [self.initial_output_dir, self.aggregated_output_dir, self.analysis_output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.knowledge_list = []
        self.cluster_info = []
        self.flattened_knowledge_list = []
        self.initial_cooccurrence = {}
        self.aggregated_cooccurrence = {}
        self.cluster_mapping = {}  # Maps original names to cluster representatives
        
        # Analysis parameters
        self.data_name = "cogatom"
        self.n_clusters = 247
        self.similarity_threshold = 0.9
        
        logger.info(f"CogAtom Cooccurrence Analyzer initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Output base: {self.output_base}")
    
    def load_knowledge_list(self):
        """Load knowledge_list from JSON file"""
        logger.info(f"Loading knowledge_list from: {self.knowledge_list_file}")
        
        if not self.knowledge_list_file.exists():
            raise FileNotFoundError(f"Knowledge list file not found: {self.knowledge_list_file}")
        
        with open(self.knowledge_list_file, 'r', encoding='utf-8') as f:
            self.knowledge_list = json.load(f)
        
        # Create flattened list for clustering index mapping
        self.flattened_knowledge_list = [point for sublist in self.knowledge_list for point in sublist]
        
        logger.info(f"Loaded {len(self.knowledge_list)} problems with {len(self.flattened_knowledge_list)} total knowledge points")
        return True
    
    def parse_cluster_line(self, line):
        """Parse cluster file line into dictionary"""
        entry = {}
        parts = line.strip().split('##')
        
        for part in parts:
            if part.startswith('[') and part.endswith(']'):
                part_content = part[1:-1].strip()
                key_value = part_content.split(': ', 1)
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    
                    # Try to convert value to appropriate type
                    if value.startswith('[') and value.endswith(']'):
                        try:
                            entry[key] = ast.literal_eval(value)
                        except (SyntaxError, ValueError) as e:
                            logger.warning(f"Failed to parse list value: {e}")
                            entry[key] = value
                    else:
                        # Try to convert to int if possible
                        try:
                            entry[key] = int(value)
                        except ValueError:
                            entry[key] = value
        
        return entry
    
    def load_cluster_info(self):
        """Load clustering information from file and create mapping"""
        logger.info(f"Loading cluster info from: {self.cluster_file}")
        
        if not self.cluster_file.exists():
            raise FileNotFoundError(f"Cluster file not found: {self.cluster_file}")
        
        self.cluster_info = []
        with open(self.cluster_file, 'r', encoding='utf-8') as f:
            for line in f:
                parsed_entry = self.parse_cluster_line(line)
                if parsed_entry:
                    self.cluster_info.append(parsed_entry)
        
        logger.info(f"Loaded {len(self.cluster_info)} cluster groups")
        
        # Create cluster mapping
        self.create_cluster_mapping()
        
        return True
    
    def create_cluster_mapping(self):
        """Create mapping from original knowledge points to cluster representatives"""
        logger.info("Creating cluster mapping...")
        
        # Initialize: each knowledge point maps to itself
        self.cluster_mapping = {}
        all_names = set()
        
        for point in self.flattened_knowledge_list:
            name = point.split(' - ')[0].strip()
            all_names.add(name)
            self.cluster_mapping[name] = name
        
        # Apply clustering: map cluster members to representatives
        clustered_count = 0
        cluster_stats = defaultdict(int)
        
        for cluster_entry in self.cluster_info:
            main_index = cluster_entry.get('main_index', 0)
            indices = cluster_entry.get('indices', [])
            
            if main_index >= len(self.flattened_knowledge_list):
                logger.warning(f"Main index {main_index} out of range")
                continue
            
            main_keyword = self.flattened_knowledge_list[main_index].split(' - ')[0].strip()
            cluster_size = len(indices)
            cluster_stats[cluster_size] += 1
            
            for index in indices:
                if index >= len(self.flattened_knowledge_list):
                    logger.warning(f"Index {index} out of range")
                    continue
                
                other_keyword = self.flattened_knowledge_list[index].split(' - ')[0].strip()
                
                if other_keyword != main_keyword:
                    self.cluster_mapping[other_keyword] = main_keyword
                    clustered_count += 1
        
        # Calculate statistics
        unique_representatives = len(set(self.cluster_mapping.values()))
        original_count = len(all_names)
        reduction = original_count - unique_representatives
        reduction_rate = (reduction / original_count) * 100 if original_count > 0 else 0
        
        logger.info(f"Cluster mapping created:")
        logger.info(f"  Original knowledge points: {original_count}")
        logger.info(f"  Unique representatives: {unique_representatives}")
        logger.info(f"  Points clustered: {clustered_count}")
        logger.info(f"  Reduction: {reduction} points ({reduction_rate:.1f}%)")
        
        logger.info(f"  Cluster size distribution:")
        for size in sorted(cluster_stats.keys()):
            count = cluster_stats[size]
            logger.info(f"    Size {size}: {count} clusters")
        
        return self.cluster_mapping
    
    def calculate_initial_cooccurrence(self):
        """Calculate initial cooccurrence statistics based on original knowledge points"""
        logger.info("Calculating initial cooccurrence statistics...")
        
        self.initial_cooccurrence = {}
        
        for k_list in tqdm(self.knowledge_list, desc="Processing problems for initial cooccurrence"):
            # Extract knowledge point names (before " - ")
            keywords = [k.split(' - ')[0].strip() for k in k_list]
            
            # Generate all pairs
            edges = list(combinations(keywords, 2))
            
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                self.initial_cooccurrence[sorted_edge] = self.initial_cooccurrence.get(sorted_edge, 0) + 1
        
        logger.info(f"Initial cooccurrence calculation completed: {len(self.initial_cooccurrence)} unique pairs")
        return self.initial_cooccurrence
    
    def aggregate_cooccurrence_relationships(self):
        """
        Aggregate cooccurrence relationships based on clustering.
        
        Key Logic:
        - For each cluster, transfer all cooccurrence relationships from cluster members 
          to the cluster representative
        - Accumulate weights when multiple members have relationships with the same knowledge point
        - Remove self-loops that may be created when cluster members had relationships with each other
        """
        logger.info("Aggregating cooccurrence relationships based on clustering...")
        
        # Start with a copy of initial cooccurrence
        self.aggregated_cooccurrence = self.initial_cooccurrence.copy()
        
        # Track aggregation statistics
        aggregation_stats = {
            'clusters_processed': 0,
            'relationships_transferred': 0,
            'edges_modified': 0,
            'edges_created': 0,
            'edges_deleted': 0,
            'self_loops_removed': 0,
            'total_weight_transferred': 0
        }
        
        # Process each cluster
        for cluster_entry in tqdm(self.cluster_info, desc="Applying clustering aggregation"):
            main_index = cluster_entry.get('main_index', 0)
            indices = cluster_entry.get('indices', [])
            
            if not indices or main_index not in indices:
                continue
            
            # Get main keyword (cluster representative)
            if main_index >= len(self.flattened_knowledge_list):
                logger.warning(f"Main index {main_index} out of range")
                continue
            
            main_keyword = self.flattened_knowledge_list[main_index].split(' - ')[0].strip()
            
            # Process each member of the cluster (except the representative itself)
            members_processed = 0
            
            for index in indices:
                if index == main_index:
                    continue  # Skip the main representative
                
                if index >= len(self.flattened_knowledge_list):
                    logger.warning(f"Index {index} out of range")
                    continue
                
                other_keyword = self.flattened_knowledge_list[index].split(' - ')[0].strip()
                
                if other_keyword == main_keyword:
                    continue  # Skip if already the same name
                
                members_processed += 1
                
                # Find all edges containing this cluster member and transfer to representative
                edges_to_process = list(self.aggregated_cooccurrence.keys())
                
                for edge in edges_to_process:
                    if other_keyword in edge:
                        # Create new edge by replacing cluster member with representative
                        new_edge_list = []
                        for kw in edge:
                            if kw == other_keyword:
                                new_edge_list.append(main_keyword)
                            else:
                                new_edge_list.append(kw)
                        
                        new_edge = tuple(sorted(new_edge_list))
                        
                        # Skip self-loops (when both points in edge become the same after clustering)
                        if len(set(new_edge)) < 2:
                            aggregation_stats['self_loops_removed'] += 1
                            del self.aggregated_cooccurrence[edge]
                            aggregation_stats['edges_deleted'] += 1
                            continue
                        
                        # Transfer relationship if edge changed
                        if new_edge != edge:
                            old_weight = self.aggregated_cooccurrence[edge]
                            aggregation_stats['total_weight_transferred'] += old_weight
                            aggregation_stats['relationships_transferred'] += 1
                            
                            # Add weight to new edge (accumulate if exists)
                            if new_edge in self.aggregated_cooccurrence:
                                self.aggregated_cooccurrence[new_edge] += old_weight
                                aggregation_stats['edges_modified'] += 1
                            else:
                                self.aggregated_cooccurrence[new_edge] = old_weight
                                aggregation_stats['edges_created'] += 1
                            
                            # Remove old edge
                            del self.aggregated_cooccurrence[edge]
                            aggregation_stats['edges_deleted'] += 1
            
            if members_processed > 0:
                aggregation_stats['clusters_processed'] += 1
        
        logger.info(f"Cooccurrence aggregation completed:")
        logger.info(f"  Clusters processed: {aggregation_stats['clusters_processed']}")
        logger.info(f"  Relationships transferred: {aggregation_stats['relationships_transferred']}")
        logger.info(f"  Edges created: {aggregation_stats['edges_created']}")
        logger.info(f"  Edges modified: {aggregation_stats['edges_modified']}")
        logger.info(f"  Edges deleted: {aggregation_stats['edges_deleted']}")
        logger.info(f"  Self-loops removed: {aggregation_stats['self_loops_removed']}")
        logger.info(f"  Total weight transferred: {aggregation_stats['total_weight_transferred']}")
        
        return aggregation_stats
    
    def print_cooccurrence_statistics(self, cooccurrence_dict, title):
        """Print detailed cooccurrence statistics"""
        logger.info(f"\n{'='*60}")
        logger.info(f"STATISTICS: {title}")
        logger.info(f"{'='*60}")
        
        if not cooccurrence_dict:
            logger.info("No cooccurrence data available")
            return {}
        
        # Calculate distribution
        distribution = Counter(cooccurrence_dict.values())
        total_pairs = len(cooccurrence_dict)
        total_occurrences = sum(cooccurrence_dict.values())
        
        logger.info(f"Total unique pairs: {total_pairs}")
        logger.info(f"Total occurrences: {total_occurrences}")
        logger.info(f"Average occurrences per pair: {total_occurrences/total_pairs:.2f}")
        
        # Calculate percentiles
        frequencies = sorted(cooccurrence_dict.values())
        if frequencies:
            p50 = frequencies[len(frequencies)//2]
            p90 = frequencies[int(len(frequencies)*0.9)]
            p95 = frequencies[int(len(frequencies)*0.95)]
            logger.info(f"Frequency percentiles - 50th: {p50}, 90th: {p90}, 95th: {p95}")
        
        logger.info(f"\nCooccurrence frequency distribution:")
        for count in sorted(distribution.keys())[:10]:  # Show top 10 frequencies
            frequency = distribution[count]
            percentage = (frequency / total_pairs) * 100
            logger.info(f"  {count} occurrences: {frequency} pairs ({percentage:.1f}%)")
        
        if len(distribution) > 10:
            logger.info(f"  ... and {len(distribution)-10} more frequency levels")
        
        # Show top and bottom pairs
        sorted_pairs = sorted(cooccurrence_dict.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"\nTop 10 most frequent pairs:")
        for i, (pair, count) in enumerate(sorted_pairs[:10]):
            logger.info(f"  {i+1}. {pair[0]} <-> {pair[1]}: {count} times")
        
        logger.info(f"\nBottom 5 least frequent pairs:")
        for i, (pair, count) in enumerate(sorted_pairs[-5:]):
            logger.info(f"  {len(sorted_pairs)-4+i}. {pair[0]} <-> {pair[1]}: {count} times")
        
        # Analyze knowledge point coverage
        all_knowledge_points = set()
        for pair in cooccurrence_dict.keys():
            all_knowledge_points.update(pair)
        
        logger.info(f"\nKnowledge point coverage:")
        logger.info(f"  Unique knowledge points in pairs: {len(all_knowledge_points)}")
        
        logger.info(f"{'='*60}")
        
        return {
            'total_pairs': total_pairs,
            'total_occurrences': total_occurrences,
            'average_occurrences': total_occurrences/total_pairs if total_pairs > 0 else 0,
            'unique_knowledge_points': len(all_knowledge_points),
            'distribution': dict(distribution),
            'top_10_pairs': sorted_pairs[:10],
            'bottom_5_pairs': sorted_pairs[-5:],
            'percentiles': {
                '50th': frequencies[len(frequencies)//2] if frequencies else 0,
                '90th': frequencies[int(len(frequencies)*0.9)] if frequencies else 0,
                '95th': frequencies[int(len(frequencies)*0.95)] if frequencies else 0
            }
        }
    
    def save_cooccurrence_data(self, cooccurrence_dict, file_path, title):
        """Save cooccurrence data to file in the format: point1##point2 - frequency"""
        logger.info(f"Saving {title} to: {file_path}")
        
        if not cooccurrence_dict:
            logger.warning(f"No data to save for {title}")
            return 0
        
        # Sort by frequency (descending)
        sorted_pairs = sorted(cooccurrence_dict.items(), key=lambda x: x[1], reverse=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for (point1, point2), frequency in sorted_pairs:
                f.write(f"{point1}##{point2} - {frequency}\n")
        
        logger.info(f"Saved {len(sorted_pairs)} cooccurrence pairs")
        return len(sorted_pairs)
    
    def analyze_clustering_impact(self, initial_stats, aggregated_stats):
        """Analyze the impact of clustering on cooccurrence relationships"""
        
        impact_analysis = {
            'pair_reduction': {
                'absolute': initial_stats['total_pairs'] - aggregated_stats['total_pairs'],
                'percentage': ((initial_stats['total_pairs'] - aggregated_stats['total_pairs']) / initial_stats['total_pairs'] * 100) if initial_stats['total_pairs'] > 0 else 0
            },
            'occurrence_change': {
                'absolute': aggregated_stats['total_occurrences'] - initial_stats['total_occurrences'],
                'percentage': ((aggregated_stats['total_occurrences'] - initial_stats['total_occurrences']) / initial_stats['total_occurrences'] * 100) if initial_stats['total_occurrences'] > 0 else 0
            },
            'knowledge_point_reduction': {
                'absolute': initial_stats['unique_knowledge_points'] - aggregated_stats['unique_knowledge_points'],
                'percentage': ((initial_stats['unique_knowledge_points'] - aggregated_stats['unique_knowledge_points']) / initial_stats['unique_knowledge_points'] * 100) if initial_stats['unique_knowledge_points'] > 0 else 0
            },
            'average_frequency_change': aggregated_stats['average_occurrences'] - initial_stats['average_occurrences']
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("CLUSTERING IMPACT ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Pair reduction: {impact_analysis['pair_reduction']['absolute']} ({impact_analysis['pair_reduction']['percentage']:.1f}%)")
        logger.info(f"Knowledge point reduction: {impact_analysis['knowledge_point_reduction']['absolute']} ({impact_analysis['knowledge_point_reduction']['percentage']:.1f}%)")
        logger.info(f"Total occurrences change: {impact_analysis['occurrence_change']['absolute']:+d} ({impact_analysis['occurrence_change']['percentage']:+.1f}%)")
        logger.info(f"Average frequency change: {impact_analysis['average_frequency_change']:+.2f}")
        logger.info(f"{'='*60}")
        
        return impact_analysis
    
    def save_comprehensive_results(self, initial_stats, aggregated_stats, aggregation_stats, impact_analysis):
        """Save comprehensive analysis results"""
        
        # 1. Save detailed statistics JSON
        stats_file = self.analysis_output_dir / f"cooccurrence_analysis_stats_{self.n_clusters}_{int(self.similarity_threshold*100)}.json"
        
        analysis_results = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'data_name': self.data_name,
                'n_clusters': self.n_clusters,
                'similarity_threshold': self.similarity_threshold,
                'total_problems': len(self.knowledge_list),
                'total_knowledge_points': len(self.flattened_knowledge_list)
            },
            'initial_cooccurrence': initial_stats,
            'aggregated_cooccurrence': aggregated_stats,
            'aggregation_process': aggregation_stats,
            'clustering_impact': impact_analysis
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Comprehensive analysis results saved to: {stats_file}")
        
        # 2. Save cluster mapping for reference
        mapping_file = self.analysis_output_dir / f"cluster_mapping_{self.n_clusters}_{int(self.similarity_threshold*100)}.json"
        
        # Create reverse mapping (representative -> members)
        reverse_mapping = defaultdict(list)
        for original, representative in self.cluster_mapping.items():
            reverse_mapping[representative].append(original)
        
        # Only include clusters with more than one member
        actual_clusters = {rep: members for rep, members in reverse_mapping.items() if len(members) > 1}
        
        mapping_data = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_original_points': len(self.cluster_mapping),
                'total_representatives': len(set(self.cluster_mapping.values())),
                'actual_clusters': len(actual_clusters)
            },
            'original_to_representative': self.cluster_mapping,
            'representative_to_members': dict(reverse_mapping),
            'actual_clusters_only': actual_clusters
        }
        
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Cluster mapping saved to: {mapping_file}")
        
        # 3. Save summary report
        summary_file = self.analysis_output_dir / f"analysis_summary_{self.n_clusters}_{int(self.similarity_threshold*100)}.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("CogAtom Cooccurrence Analysis Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_name}\n")
            f.write(f"Clusters: {self.n_clusters}\n")
            f.write(f"Similarity Threshold: {self.similarity_threshold}\n\n")
            
            f.write("Data Overview:\n")
            f.write(f"  Total Problems: {len(self.knowledge_list)}\n")
            f.write(f"  Total Knowledge Points: {len(self.flattened_knowledge_list)}\n\n")
            
            f.write("Initial Cooccurrence:\n")
            f.write(f"  Unique Pairs: {initial_stats['total_pairs']}\n")
            f.write(f"  Total Occurrences: {initial_stats['total_occurrences']}\n")
            f.write(f"  Average Frequency: {initial_stats['average_occurrences']:.2f}\n")
            f.write(f"  Unique Knowledge Points: {initial_stats['unique_knowledge_points']}\n\n")
            
            f.write("Aggregated Cooccurrence:\n")
            f.write(f"  Unique Pairs: {aggregated_stats['total_pairs']}\n")
            f.write(f"  Total Occurrences: {aggregated_stats['total_occurrences']}\n")
            f.write(f"  Average Frequency: {aggregated_stats['average_occurrences']:.2f}\n")
            f.write(f"  Unique Knowledge Points: {aggregated_stats['unique_knowledge_points']}\n\n")
            
            f.write("Clustering Impact:\n")
            f.write(f"  Pairs Reduced: {impact_analysis['pair_reduction']['absolute']} ({impact_analysis['pair_reduction']['percentage']:.1f}%)\n")
            f.write(f"  Knowledge Points Reduced: {impact_analysis['knowledge_point_reduction']['absolute']} ({impact_analysis['knowledge_point_reduction']['percentage']:.1f}%)\n")
            f.write(f"  Average Frequency Change: {impact_analysis['average_frequency_change']:+.2f}\n\n")
            
            f.write("Processing Statistics:\n")
            f.write(f"  Clusters Processed: {aggregation_stats['clusters_processed']}\n")
            f.write(f"  Relationships Transferred: {aggregation_stats['relationships_transferred']}\n")
            f.write(f"  Self-loops Removed: {aggregation_stats['self_loops_removed']}\n")
        
        logger.info(f"Analysis summary saved to: {summary_file}")
        
        return analysis_results
    
    def run_complete_analysis(self):
        """Run complete cooccurrence analysis pipeline"""
        logger.info("="*80)
        logger.info("STARTING COGATOM COOCCURRENCE ANALYSIS")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Step 1: Load data
            logger.info("\n" + "="*60)
            logger.info("STEP 1: LOADING DATA")
            logger.info("="*60)
            
            self.load_knowledge_list()
            self.load_cluster_info()
            
            # Step 2: Calculate initial cooccurrence
            logger.info("\n" + "="*60)
            logger.info("STEP 2: INITIAL COOCCURRENCE CALCULATION")
            logger.info("="*60)
            
            self.calculate_initial_cooccurrence()
            initial_stats = self.print_cooccurrence_statistics(self.initial_cooccurrence, "Initial Cooccurrence")
            
            # Save initial results
            initial_file = self.initial_output_dir / f"initial_cooccurrence_{self.n_clusters}_{int(self.similarity_threshold*100)}.txt"
            self.save_cooccurrence_data(self.initial_cooccurrence, initial_file, "initial cooccurrence data")
            
            # Step 3: Apply clustering aggregation
            logger.info("\n" + "="*60)
            logger.info("STEP 3: CLUSTERING AGGREGATION")
            logger.info("="*60)
            
            aggregation_stats = self.aggregate_cooccurrence_relationships()
            aggregated_stats = self.print_cooccurrence_statistics(self.aggregated_cooccurrence, "Aggregated Cooccurrence")
            
            # Save aggregated results
            aggregated_file = self.aggregated_output_dir / f"aggregated_cooccurrence_{self.n_clusters}_{int(self.similarity_threshold*100)}.txt"
            self.save_cooccurrence_data(self.aggregated_cooccurrence, aggregated_file, "aggregated cooccurrence data")
            
            # Step 4: Analyze clustering impact
            logger.info("\n" + "="*60)
            logger.info("STEP 4: CLUSTERING IMPACT ANALYSIS")
            logger.info("="*60)
            
            impact_analysis = self.analyze_clustering_impact(initial_stats, aggregated_stats)
            
            # Step 5: Save comprehensive results
            logger.info("\n" + "="*60)
            logger.info("STEP 5: SAVING COMPREHENSIVE RESULTS")
            logger.info("="*60)
            
            analysis_results = self.save_comprehensive_results(initial_stats, aggregated_stats, aggregation_stats, impact_analysis)
            
            # Step 6: Final summary
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("\n" + "="*80)
            logger.info("COOCCURRENCE ANALYSIS COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Total processing time: {duration:.2f} seconds")
            logger.info(f"")
            logger.info(f"📊 FINAL SUMMARY:")
            logger.info(f"  Problems analyzed: {len(self.knowledge_list)}")
            logger.info(f"  Knowledge points: {len(self.flattened_knowledge_list)}")
            logger.info(f"  Initial pairs: {initial_stats['total_pairs']}")
            logger.info(f"  Aggregated pairs: {aggregated_stats['total_pairs']}")
            logger.info(f"  Pairs reduction: {impact_analysis['pair_reduction']['absolute']} ({impact_analysis['pair_reduction']['percentage']:.1f}%)")
            logger.info(f"  Knowledge points reduction: {impact_analysis['knowledge_point_reduction']['absolute']} ({impact_analysis['knowledge_point_reduction']['percentage']:.1f}%)")
            logger.info(f"")
            logger.info(f"📁 OUTPUT FILES:")
            logger.info(f"  Initial cooccurrence: {initial_file}")
            logger.info(f"  Aggregated cooccurrence: {aggregated_file}")
            logger.info(f"  Analysis results: {self.analysis_output_dir}")
            logger.info(f"")
            logger.info(f"🚀 READY FOR GRAPH CONSTRUCTION!")
            logger.info("="*80)
            
            return {
                'success': True,
                'duration': duration,
                'initial_stats': initial_stats,
                'aggregated_stats': aggregated_stats,
                'aggregation_stats': aggregation_stats,
                'impact_analysis': impact_analysis,
                'files': {
                    'initial_cooccurrence': str(initial_file),
                    'aggregated_cooccurrence': str(aggregated_file),
                    'analysis_results': str(self.analysis_output_dir)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = CogAtomCooccurrenceAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if results['success']:
        print(f"\n✅ Analysis completed successfully in {results['duration']:.2f} seconds")
        return 0
    else:
        print(f"\n❌ Analysis failed: {results['error']}")
        return 1


if __name__ == "__main__":
    exit(main())
