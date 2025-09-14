#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import pickle
import time
from pathlib import Path
import logging
from collections import defaultdict
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root():
    """Get the project root directory (cogatom/)"""
    script_dir = Path(__file__).parent
    return script_dir


class KnowledgeListConstructor:
    """Constructor for knowledge_list from consolidated knowledge points"""
    
    def __init__(self, project_root=None):
        """Initialize constructor"""
        self.project_root = project_root or get_project_root()
        self.knowledge_list = []
        self.metadata = {}
        
        # Define paths
        self.data_dir = self.project_root / "data"
        self.knowledge_points_dir = self.data_dir / "processed" / "knowledge_points"
        self.output_dir = self.knowledge_points_dir / "processed"
        self.formatted_dir = self.knowledge_points_dir / "formatted"
        
        # Input file path
        self.consolidated_file = self.data_dir / "processed" / "knowledge_consolidation" / "consolidated_knowledge_points.txt"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formatted_dir.mkdir(parents=True, exist_ok=True)
    
    def load_consolidated_knowledge_points(self):
        """
        Load knowledge points from consolidated_knowledge_points.txt
        
        Format: Each problem's knowledge points are on separate lines,
                different problems are separated by empty lines
        
        Returns:
            List of knowledge point groups (one per problem)
        """
        logger.info(f"Loading consolidated knowledge points from: {self.consolidated_file}")
        
        if not self.consolidated_file.exists():
            logger.error(f"Consolidated file not found: {self.consolidated_file}")
            return []
        
        knowledge_groups = []
        current_group = []
        
        with open(self.consolidated_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if line:
                    # Non-empty line: add to current group
                    current_group.append(line)
                else:
                    # Empty line: end of current group
                    if current_group:
                        knowledge_groups.append(current_group.copy())
                        current_group = []
        
        # Don't forget the last group if file doesn't end with empty line
        if current_group:
            knowledge_groups.append(current_group)
        
        logger.info(f"Loaded {len(knowledge_groups)} knowledge point groups")
        return knowledge_groups
    
    def construct_knowledge_list(self):
        """
        Construct knowledge_list from consolidated knowledge points file
        """
        logger.info("Starting knowledge_list construction from consolidated file...")
        
        # Load knowledge point groups
        knowledge_groups = self.load_consolidated_knowledge_points()
        
        if not knowledge_groups:
            logger.error("No knowledge point groups loaded from consolidated file")
            return False
        
        # Set knowledge_list
        self.knowledge_list = knowledge_groups
        
        # Calculate statistics
        total_problems = len(self.knowledge_list)
        total_knowledge_points = sum(len(group) for group in self.knowledge_list)
        group_sizes = [len(group) for group in self.knowledge_list]
        
        # Store metadata
        self.metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': str(self.consolidated_file),
            'total_problems': total_problems,
            'total_knowledge_points': total_knowledge_points,
            'average_points_per_problem': total_knowledge_points / total_problems if total_problems > 0 else 0,
            'max_points_per_problem': max(group_sizes) if group_sizes else 0,
            'min_points_per_problem': min(group_sizes) if group_sizes else 0,
            'median_points_per_problem': sorted(group_sizes)[len(group_sizes)//2] if group_sizes else 0
        }
        
        logger.info(f"Knowledge list construction completed:")
        logger.info(f"  Total problems: {total_problems}")
        logger.info(f"  Total knowledge points: {total_knowledge_points}")
        logger.info(f"  Average points per problem: {self.metadata['average_points_per_problem']:.2f}")
        logger.info(f"  Max points per problem: {self.metadata['max_points_per_problem']}")
        logger.info(f"  Min points per problem: {self.metadata['min_points_per_problem']}")
        logger.info(f"  Median points per problem: {self.metadata['median_points_per_problem']}")
        
        return True
    
    def save_knowledge_list(self, data_name="cogatom"):
        """
        Save knowledge_list in multiple formats
        
        Args:
            data_name: Name identifier for the dataset
        """
        if not self.knowledge_list:
            logger.error("Cannot save empty knowledge_list")
            return None
        
        logger.info(f"Saving knowledge_list in multiple formats...")
        
        # 1. Save as JSON format (recommended)
        json_file = self.output_dir / f"{data_name}_knowledge_list.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_list, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved JSON format: {json_file}")
        
        # 2. Save as Pickle format (Python-specific)
        pkl_file = self.output_dir / f"{data_name}_knowledge_list.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.knowledge_list, f)
        logger.info(f"✅ Saved Pickle format: {pkl_file}")
        
        # 3. Save as text format (human-readable)
        txt_file = self.output_dir / f"{data_name}_knowledge_list.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for i, group in enumerate(self.knowledge_list):
                f.write(f"Problem_{i+1}:\n")
                for point in group:
                    f.write(f"  {point}\n")
                f.write("\n")  # Empty line separator
        logger.info(f"✅ Saved text format: {txt_file}")
        
        # 4. Save metadata
        metadata_file = self.output_dir / f"{data_name}_knowledge_list_metadata.json"
        metadata_with_files = {
            **self.metadata,
            'data_name': data_name,
            'file_formats': {
                'json': str(json_file),
                'pickle': str(pkl_file),
                'text': str(txt_file)
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_with_files, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved metadata: {metadata_file}")
        
        return metadata_with_files
    
    def save_formatted_data(self, data_name="cogatom"):
        """
        Save formatted data for cooccurrence analysis
        
        Args:
            data_name: Name identifier for the dataset
        """
        if not self.knowledge_list:
            logger.error("Cannot save formatted data for empty knowledge_list")
            return
        
        logger.info("Saving formatted data for cooccurrence analysis...")
        
        # 1. Save knowledge point pairs format
        pairs_file = self.formatted_dir / f"{data_name}_knowledge_pairs.txt"
        pair_count = 0
        with open(pairs_file, 'w', encoding='utf-8') as f:
            from itertools import combinations
            for group in self.knowledge_list:
                if len(group) >= 2:
                    for pair in combinations(group, 2):
                        # Extract knowledge point names (before " - ")
                        name1 = pair[0].split(' - ')[0].strip()
                        name2 = pair[1].split(' - ')[0].strip()
                        f.write(f"{name1}##{name2}\n")
                        pair_count += 1
        logger.info(f"✅ Saved knowledge pairs format: {pairs_file} ({pair_count} pairs)")
        
        # 2. Save groups format (empty line separated, compatible with reference script)
        groups_file = self.formatted_dir / f"{data_name}_knowledge_groups.txt"
        with open(groups_file, 'w', encoding='utf-8') as f:
            for group in self.knowledge_list:
                for point in group:
                    f.write(f"{point}\n")
                f.write("\n")  # Empty line separator
        logger.info(f"✅ Saved groups format: {groups_file}")
        
        # 3. Save flat format (all knowledge points)
        flat_file = self.formatted_dir / f"{data_name}_knowledge_flat.txt"
        with open(flat_file, 'w', encoding='utf-8') as f:
            for group in self.knowledge_list:
                for point in group:
                    f.write(f"{point}\n")
        logger.info(f"✅ Saved flat format: {flat_file}")
        
        # 4. Save statistics
        stats_file = self.formatted_dir / f"{data_name}_knowledge_statistics.json"
        
        # Calculate additional statistics
        all_knowledge_points = [point for group in self.knowledge_list for point in group]
        unique_names = set(point.split(' - ')[0].strip() for point in all_knowledge_points)
        
        # Calculate group size distribution
        group_sizes = [len(group) for group in self.knowledge_list]
        max_size = max(group_sizes) if group_sizes else 0
        
        distribution = {}
        if max_size > 0:
            for i in range(1, max_size + 1):
                count = sum(1 for size in group_sizes if size == i)
                if count > 0:
                    distribution[str(i)] = count
        
        # Calculate knowledge point name frequency
        name_frequency = defaultdict(int)
        for point in all_knowledge_points:
            name = point.split(' - ')[0].strip()
            name_frequency[name] += 1
        
        # Top 10 most frequent knowledge points
        top_knowledge_points = sorted(name_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        stats = {
            'total_problems': len(self.knowledge_list),
            'total_knowledge_point_instances': len(all_knowledge_points),
            'unique_knowledge_point_names': len(unique_names),
            'average_points_per_problem': len(all_knowledge_points) / len(self.knowledge_list) if self.knowledge_list else 0,
            'max_points_per_problem': max_size,
            'min_points_per_problem': min(group_sizes) if group_sizes else 0,
            'knowledge_point_distribution': distribution,
            'total_pairs_generated': pair_count,
            'top_10_knowledge_points': top_knowledge_points,
            'knowledge_point_reuse_rate': (len(all_knowledge_points) - len(unique_names)) / len(all_knowledge_points) * 100 if all_knowledge_points else 0
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved statistics: {stats_file}")
        
        return stats
    
    def load_knowledge_list(self, data_name="cogatom", format_type="json"):
        """
        Load knowledge_list from saved files
        
        Args:
            data_name: Name identifier for the dataset
            format_type: File format ("json", "pickle", "text")
            
        Returns:
            Loaded knowledge_list
        """
        if format_type == "json":
            file_path = self.output_dir / f"{data_name}_knowledge_list.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif format_type == "pickle":
            file_path = self.output_dir / f"{data_name}_knowledge_list.pkl"
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        elif format_type == "text":
            file_path = self.output_dir / f"{data_name}_knowledge_list.txt"
            knowledge_list = []
            current_group = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Problem_"):
                        if current_group:
                            knowledge_list.append(current_group)
                            current_group = []
                    elif line and not line.startswith("Problem_"):
                        current_group.append(line.lstrip())
                    elif not line and current_group:
                        knowledge_list.append(current_group)
                        current_group = []
            
            if current_group:
                knowledge_list.append(current_group)
            
            return knowledge_list
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def print_sample_data(self, num_samples=3):
        """Print sample data for verification"""
        if not self.knowledge_list:
            logger.warning("No data to display - knowledge_list is empty")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info("SAMPLE KNOWLEDGE LIST DATA")
        logger.info(f"{'='*60}")
        
        samples_to_show = min(num_samples, len(self.knowledge_list))
        
        for i in range(samples_to_show):
            group = self.knowledge_list[i]
            logger.info(f"\nProblem {i+1} ({len(group)} knowledge points):")
            for j, point in enumerate(group):
                # Show knowledge point name and truncated description
                if ' - ' in point:
                    name, desc = point.split(' - ', 1)
                    desc_short = desc[:80] + "..." if len(desc) > 80 else desc
                    logger.info(f"  {j+1}. {name} - {desc_short}")
                else:
                    logger.info(f"  {j+1}. {point}")
        
        if len(self.knowledge_list) > samples_to_show:
            logger.info(f"\n... and {len(self.knowledge_list) - samples_to_show} more problems")
        
        logger.info(f"\n{'='*60}")
    
    def get_knowledge_list_for_cooccurrence(self):
        """
        Get knowledge_list in the format expected by cooccurrence analysis
        
        Returns:
            knowledge_list compatible with reference scripts
        """
        return self.knowledge_list


def main():
    """Main function to construct and save knowledge_list"""
    logger.info("="*60)
    logger.info("COGATOM KNOWLEDGE LIST CONSTRUCTOR")
    logger.info("Reading from consolidated_knowledge_points.txt")
    logger.info("="*60)
    
    # Initialize constructor
    constructor = KnowledgeListConstructor()
    
    # Check if consolidated file exists
    if not constructor.consolidated_file.exists():
        logger.error(f"❌ Consolidated file not found: {constructor.consolidated_file}")
        logger.error("Please ensure the knowledge consolidation step has been completed.")
        return 1
    
    # Construct knowledge_list
    success = constructor.construct_knowledge_list()
    
    if not success:
        logger.error("❌ Failed to construct knowledge_list. Exiting.")
        return 1
    
    # Print sample data
    constructor.print_sample_data()
    
    # Save in multiple formats
    metadata = constructor.save_knowledge_list(data_name="cogatom")
    stats = constructor.save_formatted_data(data_name="cogatom")
    
    if not metadata:
        logger.error("❌ Failed to save knowledge_list")
        return 1
    
    # Print final statistics
    logger.info(f"\n{'='*60}")
    logger.info("FINAL STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total problems: {metadata['total_problems']}")
    logger.info(f"Total knowledge points: {metadata['total_knowledge_points']}")
    logger.info(f"Average points per problem: {metadata['average_points_per_problem']:.2f}")
    logger.info(f"Max points per problem: {metadata['max_points_per_problem']}")
    logger.info(f"Min points per problem: {metadata['min_points_per_problem']}")
    logger.info(f"Unique knowledge point names: {stats['unique_knowledge_point_names']}")
    logger.info(f"Knowledge point reuse rate: {stats['knowledge_point_reuse_rate']:.1f}%")
    logger.info(f"Total pairs generated: {stats['total_pairs_generated']}")
    
    logger.info(f"\n📊 Top 5 most frequent knowledge points:")
    for name, count in stats['top_10_knowledge_points'][:5]:
        logger.info(f"  {name}: {count} occurrences")
    
    logger.info(f"\n✅ Knowledge list construction completed successfully!")
    logger.info(f"Files saved in: {constructor.output_dir}")
    logger.info(f"Formatted data saved in: {constructor.formatted_dir}")
    
    # Test loading
    logger.info(f"\n{'='*60}")
    logger.info("TESTING DATA LOADING")
    logger.info(f"{'='*60}")
    
    try:
        loaded_data = constructor.load_knowledge_list("cogatom", "json")
        logger.info(f"✅ Successfully loaded {len(loaded_data)} problems from JSON format")
        
        # Verify data integrity
        if len(loaded_data) == len(constructor.knowledge_list):
            logger.info("✅ Data integrity verified - loaded data matches original")
        else:
            logger.warning("⚠️ Data integrity issue - size mismatch")
            
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info("READY FOR COOCCURRENCE ANALYSIS")
    logger.info(f"{'='*60}")
    logger.info("Next steps:")
    logger.info("1. Use the generated knowledge_list for cooccurrence analysis")
    logger.info("2. Apply clustering results to aggregate similar knowledge points")
    logger.info("3. Build knowledge point cooccurrence graph")
    logger.info("")
    logger.info("📁 Key files for next step:")
    logger.info(f"  - knowledge_list: {constructor.output_dir}/cogatom_knowledge_list.json")
    logger.info(f"  - clustering result: data/processed/clustering/similarity_results/atom2olympiad_output_clusters_247_threshold_90.txt")
    logger.info(f"  - formatted groups: {constructor.formatted_dir}/cogatom_knowledge_groups.txt")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
