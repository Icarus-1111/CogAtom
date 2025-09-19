#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
import subprocess
import sys

class DataProcessor:
    """Complete data processing pipeline for CogAtom"""
    
    def __init__(self):
        self.base_path = Path("data/raw/download_data/huggingface.co/datasets")
        self.output_with_docid = Path("data/processed/with_docid")
        self.output_for_extraction = Path("data/processed/for_extraction")
        
        # Dataset configuration - Only GSM8K_train
        self.datasets = [
            {
                "name": "GSM8K_train",
                "type": "parquet",
                "path": self.base_path / "openai/gsm8k/main/train-00000-of-00001.parquet"
            }
        ]
        
        # Configuration
        self.uid_fields = ["unique_id", "ID", "id"]
        self.question_keys = ["question", "problem", "Problem"]
        
    def print_sep(self, title):
        """Print separator with title"""
        print("\n" + "="*60)
        print(title)
        print("="*60)
    
    def get_file_size(self, filepath):
        """Get file size in MB"""
        try:
            size_bytes = os.path.getsize(filepath)
            size_mb = size_bytes / 1024 / 1024
            return f"{size_mb:.2f} MB"
        except Exception as e:
            return "N/A"
    
    def explore_parquet(self, filepath, n_sample=3):
        """Explore parquet file structure and content"""
        try:
            df = pd.read_parquet(filepath, engine='pyarrow')
            n_rows = len(df)
            columns = list(df.columns)
            sample = df.head(n_sample).to_dict(orient='records')
            return {
                "n_rows": n_rows,
                "columns": columns,
                "sample": sample
            }
        except Exception as e:
            print(f"[Error] Failed to read parquet: {filepath} - {e}")
            return {"n_rows": 0, "columns": [], "sample": []}
    
    def explore_jsonl(self, filepath, n_sample=3):
        """Explore JSONL file structure and content"""
        n_rows = 0
        columns = set()
        sample = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        continue
                    n_rows += 1
                    if n_rows <= n_sample:
                        sample.append(obj)
                    columns.update(obj.keys())
            return {
                "n_rows": n_rows,
                "columns": list(columns),
                "sample": sample
            }
        except Exception as e:
            print(f"[Error] Failed to read jsonl: {filepath} - {e}")
            return {"n_rows": 0, "columns": [], "sample": []}
    
    def step1_explore_datasets(self):
        """Step 1: Explore dataset structure"""
        self.print_sep("Step 1: Dataset Structure Exploration")
        
        all_info = []
        
        for dataset in self.datasets:
            print(f"\n--- Dataset: {dataset['name']} ---")
            filepath = dataset['path']
            print(f"Path: {filepath}")
            print(f"Type: {dataset['type']}")
            print(f"Size: {self.get_file_size(filepath)}")
            
            if not filepath.exists():
                print(f"[Warning] File not found: {filepath}")
                continue
                
            if dataset['type'] == "parquet":
                info = self.explore_parquet(filepath)
            else:
                info = self.explore_jsonl(filepath)
                
            print(f"Rows: {info['n_rows']}")
            print(f"Columns: {info['columns']}")
            print("Sample data:")
            for i, ex in enumerate(info['sample']):
                print(f"  Example {i+1}: {json.dumps(ex, ensure_ascii=False, indent=2)}")
            
            # Collect summary info
            all_info.append({
                "name": dataset['name'],
                "type": dataset['type'],
                "size": self.get_file_size(filepath),
                "n_rows": info['n_rows'],
                "columns": info['columns']
            })
        
        # Summary table
        print(f"\n--- Dataset Summary ---")
        if all_info:
            table = []
            for info in all_info:
                table.append([
                    info['name'],
                    info['type'],
                    info['size'],
                    info['n_rows'],
                    ", ".join(info['columns'])
                ])
            
            headers = ["Dataset", "Type", "Size", "Rows", "Columns"]
            print(tabulate(table, headers=headers, tablefmt="grid"))
        
        return all_info
    
    def build_doc_id(self, dataset_name, data, line_no):
        """Generate doc_id based on unique identifier or line number"""
        for key in self.uid_fields:
            if key in data:
                return f"{{{dataset_name}}}{{{data[key]}}}"
        return f"{{{dataset_name}}}{{{line_no}}}"
    
    def process_parquet_with_docid(self, dataset_name, input_path, output_path):
        """Process parquet file and generate JSONL with doc_id"""
        print(f"Processing Parquet: {input_path}")
        df = pd.read_parquet(input_path, engine="pyarrow")
        n_rows = len(df)
        
        used_uid_field = None
        doc_ids = set()
        samples = []
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            for i, row in tqdm(df.iterrows(), total=n_rows, desc=f"{dataset_name}"):
                data = dict(row)
                doc_id = self.build_doc_id(dataset_name, data, i)
                
                if used_uid_field is None:
                    for key in self.uid_fields:
                        if key in data:
                            used_uid_field = key
                            break
                    if used_uid_field is None:
                        used_uid_field = "line_number"
                
                data = {"doc_id": doc_id, **data}
                doc_ids.add(doc_id)
                
                if i < 2:
                    samples.append(data)
                    
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        print(f"  → {n_rows} rows, {len(doc_ids)} unique doc_ids, ID field: {used_uid_field}")
        return n_rows, len(doc_ids), used_uid_field
    
    def process_jsonl_with_docid(self, dataset_name, input_path, output_path):
        """Process JSONL file and generate JSONL with doc_id"""
        print(f"Processing JSONL: {input_path}")
        n_rows = 0
        doc_ids = set()
        used_uid_field = None
        samples = []
        
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for i, line in enumerate(tqdm(fin, desc=dataset_name)):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    continue
                    
                doc_id = self.build_doc_id(dataset_name, data, n_rows)
                
                if used_uid_field is None:
                    for key in self.uid_fields:
                        if key in data:
                            used_uid_field = key
                            break
                    if used_uid_field is None:
                        used_uid_field = "line_number"
                
                data = {"doc_id": doc_id, **data}
                doc_ids.add(doc_id)
                n_rows += 1
                
                if n_rows <= 2:
                    samples.append(data)
                    
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        print(f"  → {n_rows} rows, {len(doc_ids)} unique doc_ids, ID field: {used_uid_field}")
        return n_rows, len(doc_ids), used_uid_field
    
    def step2_add_docid(self):
        """Step 2: Add document IDs"""
        self.print_sep("Step 2: Adding Document IDs")
        
        # Create output directory
        os.makedirs(self.output_with_docid, exist_ok=True)
        
        stats = []
        
        for dataset in self.datasets:
            name = dataset["name"]
            dtype = dataset["type"]
            input_path = dataset["path"]
            output_path = self.output_with_docid / f"{name}_with_docid.jsonl"
            
            if not input_path.exists():
                print(f"[ERROR] Input file not found: {input_path}")
                continue
            
            if dtype == "parquet":
                n, n_unique, uid_field = self.process_parquet_with_docid(name, input_path, output_path)
            elif dtype == "jsonl":
                n, n_unique, uid_field = self.process_jsonl_with_docid(name, input_path, output_path)
            else:
                print(f"[ERROR] Unsupported data type: {dtype}")
                continue
                
            stats.append({
                "dataset": name,
                "total_rows": n,
                "unique_doc_ids": n_unique,
                "id_field": uid_field,
                "output_file": str(output_path)
            })
        
        print(f"\nGenerated {len(stats)} files with doc_ids")
        return stats
    
    def clean_question(self, q):
        """Clean question text - remove newlines, tabs, normalize spaces"""
        if not isinstance(q, str):
            return ""
        q = q.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        q = ' '.join(q.split())
        return q
    
    def process_file_for_extraction(self, input_file):
        """Process single JSONL file to extraction format"""
        input_path = Path(input_file)
        basename = input_path.stem
        output_file = self.output_for_extraction / f"{basename}_for_extraction.txt"
        
        n_total = 0
        n_no_question = 0
        samples = []

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"Processing {input_path.name}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    continue
                    
                doc_id = data.get("doc_id", None)
                if not doc_id:
                    continue
                
                # Extract question text
                question = None
                for key in self.question_keys:
                    if key in data and data[key] and isinstance(data[key], str):
                        question = data[key]
                        break
                
                if not question:
                    n_no_question += 1
                    continue
                
                question = self.clean_question(question)
                
                # Output format: doc_id##@@Bquestion
                out_line = f"{doc_id}##@@B{question}"
                fout.write(out_line + "\n")
                n_total += 1
                
                if len(samples) < 2:
                    samples.append(out_line)
        
        print(f"  → {input_path.name}: {n_total} samples, {n_no_question} without questions")
        return n_total, n_no_question, samples
    
    def step3_format_for_extraction(self):
        """Step 3: Format data for knowledge extraction"""
        self.print_sep("Step 3: Formatting for Knowledge Extraction")
        
        # Create output directory
        os.makedirs(self.output_for_extraction, exist_ok=True)
        
        # Process all JSONL files
        if not self.output_with_docid.exists():
            print(f"[ERROR] Input directory not found: {self.output_with_docid}")
            return []
        
        jsonl_files = list(self.output_with_docid.glob("*.jsonl"))
        if not jsonl_files:
            print(f"[ERROR] No JSONL files found in {self.output_with_docid}")
            return []
        
        results = []
        for jsonl_file in jsonl_files:
            n_total, n_no_question, samples = self.process_file_for_extraction(jsonl_file)
            results.append({
                "file": jsonl_file.name,
                "total_samples": n_total,
                "no_question": n_no_question
            })
        
        print(f"\nGenerated {len(results)} extraction files")
        return results
    
    def count_lines(self, filepath):
        """Count lines in a text file"""
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count
    
    def step4_count_lines(self):
        """Step 4: Count lines in processed files"""
        self.print_sep("Step 4: Line Count Statistics")
        
        if not self.output_for_extraction.exists():
            print(f"Directory not found: {self.output_for_extraction}")
            return []
        
        # Find all text files
        txt_files = list(self.output_for_extraction.glob("*.txt"))
        if not txt_files:
            print("No .txt files found")
            return []
        
        results = []
        total_lines = 0
        
        for txt_file in txt_files:
            line_count = self.count_lines(txt_file)
            results.append((txt_file.name, line_count))
            total_lines += line_count
        
        # Sort by filename
        results.sort()
        
        # Print results
        print(f"{'Filename':<50} {'Lines':>10}")
        print("-" * 62)
        for filename, count in results:
            print(f"{filename:<50} {count:>10}")
        
        print("-" * 62)
        print(f"{'Total files:':<50} {len(results):>10}")
        print(f"{'Total lines:':<50} {total_lines:>10}")
        
        return results
    
    def run_complete_pipeline(self):
        """Run the complete data processing pipeline"""
        print("CogAtom Data Processing Pipeline - GSM8K Train Only")
        print("="*60)
        
        try:
            # Step 1: Explore datasets
            dataset_info = self.step1_explore_datasets()
            
            # Step 2: Add doc_ids
            docid_stats = self.step2_add_docid()
            
            # Step 3: Format for extraction
            extraction_results = self.step3_format_for_extraction()
            
            # Step 4: Count lines
            line_counts = self.step4_count_lines()
            
            # Final summary
            self.print_sep("Processing Complete!")
            print("Summary:")
            print(f"  • Processed GSM8K training dataset")
            print(f"  • Generated {len(docid_stats)} files with doc_ids")
            print(f"  • Created {len(extraction_results)} extraction files")
            print(f"  • Total processed lines: {sum(count for _, count in line_counts)}")
            
            print("\nOutput directories:")
            print(f"  • With doc_ids: {self.output_with_docid}")
            print(f"  • For extraction: {self.output_for_extraction}")
            
            print("\nOutput files:")
            print(f"  • GSM8K_train_with_docid.jsonl")
            print(f"  • GSM8K_train_with_docid_for_extraction.txt")
            
            print("\nNext steps:")
            print("  1. Check the extraction file: data/processed/for_extraction/GSM8K_train_with_docid_for_extraction.txt")
            print("  2. Run knowledge extraction: python knowledge_extraction.py")
            
            return True
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    processor = DataProcessor()
    success = processor.run_complete_pipeline()
    
    if success:
        print("\n✅ GSM8K training data processing completed successfully!")
        print("📁 Ready for knowledge extraction with ~7,473 math problems")
    else:
        print("❌ Data processing failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
