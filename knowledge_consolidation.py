#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeConsolidator:
    """Consolidates knowledge points from multiple dataset extraction results"""
    
    def __init__(self, input_dir: str = "data/processed/knowledge_points", 
                 output_dir: str = "data/processed/knowledge_consolidation"):
        """
        Initialize the consolidator
        
        Args:
            input_dir: Directory containing knowledge point extraction results
            output_dir: Directory to save consolidated results
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.knowledge_groups = []  # List of knowledge point groups (one per problem)
        self.knowledge_dict = {}    # Mapping from knowledge point name to description
        self.dataset_stats = {}     # Statistics for each dataset
        self.processing_errors = [] # Track processing errors
        
    def find_dataset_directories(self) -> List[Path]:
        """Find all dataset directories in the input directory"""
        dataset_dirs = []
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return dataset_dirs
            
        for item in self.input_dir.iterdir():
            if item.is_dir() and item.name.endswith("_gpt4o_output_knowledge_points"):
                dataset_dirs.append(item)
                
        logger.info(f"Found {len(dataset_dirs)} dataset directories")
        return dataset_dirs
    
    def extract_dataset_name(self, dir_path: Path) -> str:
        """Extract dataset name from directory path"""
        dir_name = dir_path.name
        # Remove the suffix "_gpt4o_output_knowledge_points"
        if dir_name.endswith("_gpt4o_output_knowledge_points"):
            return dir_name[:-len("_gpt4o_output_knowledge_points")]
        return dir_name
    
    def parse_knowledge_points_file(self, file_path: Path) -> Tuple[List[List[str]], Dict[str, str], Dict]:
        """
        Parse a knowledge points JSONL file
        
        Args:
            file_path: Path to the knowledge points JSONL file
            
        Returns:
            Tuple of (knowledge_groups, knowledge_dict, stats)
        """
        knowledge_groups = []
        knowledge_dict = {}
        stats = {
            'total_lines': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'total_knowledge_points': 0,
            'unique_knowledge_points': 0
        }
        
        if not file_path.exists():
            logger.warning(f"Knowledge points file not found: {file_path}")
            return knowledge_groups, knowledge_dict, stats
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    stats['total_lines'] += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        doc_id = data.get('doc_id', f'unknown_{line_num}')
                        knowledge_points = data.get('knowledge_points', {})
                        
                        if not knowledge_points:
                            continue
                            
                        # Extract knowledge points for this problem
                        problem_kps = []
                        for kp_id, kp_text in knowledge_points.items():
                            if ' - ' in kp_text:
                                # Standard format: "name - description"
                                name, description = kp_text.split(' - ', 1)
                                name = name.strip()
                                description = description.strip()
                            else:
                                # Fallback: use entire text as name
                                name = kp_text.strip()
                                description = ""
                                
                            if name:
                                problem_kps.append(kp_text.strip())
                                knowledge_dict[name] = description
                                stats['total_knowledge_points'] += 1
                        
                        if problem_kps:
                            knowledge_groups.append(problem_kps)
                            stats['successful_parses'] += 1
                            
                    except json.JSONDecodeError as e:
                        stats['failed_parses'] += 1
                        error_msg = f"JSON decode error in {file_path}:{line_num}: {e}"
                        logger.warning(error_msg)
                        self.processing_errors.append(error_msg)
                        continue
                    except Exception as e:
                        stats['failed_parses'] += 1
                        error_msg = f"Processing error in {file_path}:{line_num}: {e}"
                        logger.warning(error_msg)
                        self.processing_errors.append(error_msg)
                        continue
                        
        except Exception as e:
            error_msg = f"Failed to read file {file_path}: {e}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
            
        stats['unique_knowledge_points'] = len(set(knowledge_dict.keys()))
        return knowledge_groups, knowledge_dict, stats
    
    def consolidate_all_datasets(self) -> None:
        """Consolidate knowledge points from all datasets"""
        logger.info("Starting knowledge consolidation process")
        
        dataset_dirs = self.find_dataset_directories()
        if not dataset_dirs:
            logger.error("No dataset directories found")
            return
            
        overall_stats = {
            'total_problems': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_knowledge_points': 0,
            'unique_knowledge_points': 0
        }
        
        for dataset_dir in dataset_dirs:
            dataset_name = self.extract_dataset_name(dataset_dir)
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Find knowledge points file
            kp_files = list(dataset_dir.glob("*_knowledge_points.jsonl"))
            if not kp_files:
                logger.warning(f"No knowledge points file found in {dataset_dir}")
                continue
                
            kp_file = kp_files[0]  # Take the first match
            groups, kp_dict, stats = self.parse_knowledge_points_file(kp_file)
            
            # Merge results
            self.knowledge_groups.extend(groups)
            self.knowledge_dict.update(kp_dict)
            
            # Calculate dataset statistics
            dataset_stats = {
                'total_problems': stats['total_lines'],
                'successful_extractions': stats['successful_parses'],
                'failed_extractions': stats['failed_parses'],
                'total_knowledge_points': stats['total_knowledge_points'],
                'unique_knowledge_points': stats['unique_knowledge_points']
            }
            
            self.dataset_stats[dataset_name] = dataset_stats
            
            # Update overall statistics
            overall_stats['total_problems'] += dataset_stats['total_problems']
            overall_stats['successful_extractions'] += dataset_stats['successful_extractions']
            overall_stats['failed_extractions'] += dataset_stats['failed_extractions']
            overall_stats['total_knowledge_points'] += dataset_stats['total_knowledge_points']
            
            logger.info(f"Dataset {dataset_name}: {len(groups)} problem groups, "
                       f"{stats['unique_knowledge_points']} unique knowledge points")
        
        # Calculate overall unique knowledge points
        overall_stats['unique_knowledge_points'] = len(self.knowledge_dict)
        self.dataset_stats['overall'] = overall_stats
        
        logger.info(f"Consolidation complete: {len(self.knowledge_groups)} total problem groups, "
                   f"{overall_stats['unique_knowledge_points']} unique knowledge points")
    
    def save_consolidated_knowledge_points(self) -> None:
        """Save consolidated knowledge points to text file"""
        output_file = self.output_dir / "consolidated_knowledge_points.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, group in enumerate(self.knowledge_groups):
                    # Write knowledge points for this problem
                    for kp in group:
                        f.write(f"{kp}\n")
                    
                    # Add blank line between groups (except for the last group)
                    if i < len(self.knowledge_groups) - 1:
                        f.write("\n")
                        
            logger.info(f"Saved consolidated knowledge points to {output_file}")
            
        except Exception as e:
            error_msg = f"Failed to save consolidated knowledge points: {e}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
    
    def save_knowledge_dictionary(self) -> None:
        """Save knowledge point dictionary to JSON file"""
        output_file = self.output_dir / "knowledge_point_dictionary.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_dict, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved knowledge dictionary to {output_file}")
            
        except Exception as e:
            error_msg = f"Failed to save knowledge dictionary: {e}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
    
    def save_dataset_statistics(self) -> None:
        """Save dataset statistics to JSON file"""
        output_file = self.output_dir / "dataset_statistics.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset_stats, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved dataset statistics to {output_file}")
            
        except Exception as e:
            error_msg = f"Failed to save dataset statistics: {e}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
    
    def save_processing_log(self) -> None:
        """Save processing log and errors"""
        output_file = self.output_dir / "processing_log.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Knowledge Consolidation Processing Log\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total problem groups processed: {len(self.knowledge_groups)}\n")
                f.write(f"Total unique knowledge points: {len(self.knowledge_dict)}\n")
                f.write(f"Total processing errors: {len(self.processing_errors)}\n\n")
                
                if self.processing_errors:
                    f.write("Processing Errors:\n")
                    f.write("-" * 20 + "\n")
                    for error in self.processing_errors:
                        f.write(f"{error}\n")
                    f.write("\n")
                
                f.write("Dataset Statistics:\n")
                f.write("-" * 20 + "\n")
                for dataset, stats in self.dataset_stats.items():
                    f.write(f"{dataset}:\n")
                    for key, value in stats.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                    
            logger.info(f"Saved processing log to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save processing log: {e}")
    
    def run_consolidation(self) -> bool:
        """
        Run the complete consolidation process
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Consolidate all datasets
            self.consolidate_all_datasets()
            
            if not self.knowledge_groups:
                logger.error("No knowledge groups found after consolidation")
                return False
            
            # Step 2: Save all outputs
            self.save_consolidated_knowledge_points()
            self.save_knowledge_dictionary()
            self.save_dataset_statistics()
            self.save_processing_log()
            
            # Step 3: Print summary
            self.print_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Consolidation process failed: {e}")
            return False
    
    def print_summary(self) -> None:
        """Print consolidation summary"""
        print("\n" + "=" * 60)
        print("Knowledge Consolidation Summary")
        print("=" * 60)
        
        overall_stats = self.dataset_stats.get('overall', {})
        print(f"Total problems processed: {overall_stats.get('total_problems', 0)}")
        print(f"Successful extractions: {overall_stats.get('successful_extractions', 0)}")
        print(f"Failed extractions: {overall_stats.get('failed_extractions', 0)}")
        print(f"Total knowledge points: {overall_stats.get('total_knowledge_points', 0)}")
        print(f"Unique knowledge points: {overall_stats.get('unique_knowledge_points', 0)}")
        
        if overall_stats.get('total_problems', 0) > 0:
            success_rate = (overall_stats.get('successful_extractions', 0) / 
                          overall_stats.get('total_problems', 1)) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"\nProcessing errors: {len(self.processing_errors)}")
        
        print(f"\nOutput files saved to: {self.output_dir}")
        print("- consolidated_knowledge_points.txt")
        print("- knowledge_point_dictionary.json")
        print("- dataset_statistics.json")
        print("- processing_log.txt")
        
        print("\nPer-dataset breakdown:")
        for dataset, stats in self.dataset_stats.items():
            if dataset != 'overall':
                print(f"  {dataset}: {stats['successful_extractions']}/{stats['total_problems']} problems")


def main():
    """Main function for direct execution"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python knowledge_consolidation.py")
        print("Consolidates knowledge points from extraction results")
        return
    
    # Check if input directory exists
    input_dir = "data/processed/knowledge_points"
    if not Path(input_dir).exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please run knowledge extraction first")
        return
    
    # Run consolidation
    consolidator = KnowledgeConsolidator()
    success = consolidator.run_consolidation()
    
    if success:
        print("\n✅ Knowledge consolidation completed successfully!")
    else:
        print("\n❌ Knowledge consolidation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
