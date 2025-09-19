# cogatom/knowledge_extraction.py

import os
import json
import re
import time
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 延迟导入标志
INTERNAL_API_AVAILABLE = None
STANDARD_API_AVAILABLE = None
InternalOpenAIClient = None

def _lazy_import_apis():
    """延迟导入API客户端，避免在模块加载时阻塞"""
    global INTERNAL_API_AVAILABLE, STANDARD_API_AVAILABLE, InternalOpenAIClient
    
    if INTERNAL_API_AVAILABLE is None:
        try:
            print("[DEBUG] Attempting to import internal API...")
            from open_ai_utils import OpenAIClient as _InternalOpenAIClient
            InternalOpenAIClient = _InternalOpenAIClient
            INTERNAL_API_AVAILABLE = True
            print("[INFO] Internal API client imported successfully")
        except ImportError as e:
            INTERNAL_API_AVAILABLE = False
            print(f"[INFO] Internal API client not found: {e}")
        except Exception as e:
            INTERNAL_API_AVAILABLE = False
            print(f"[ERROR] Internal API import failed: {e}")
    
    if STANDARD_API_AVAILABLE is None:
        try:
            print("[DEBUG] Attempting to import standard OpenAI API...")
            import openai
            STANDARD_API_AVAILABLE = True
            print("[INFO] Standard OpenAI API available")
        except ImportError:
            STANDARD_API_AVAILABLE = False
            print("[INFO] Standard OpenAI API not available")
    
    return INTERNAL_API_AVAILABLE, STANDARD_API_AVAILABLE

# Core functions from original implementation (UNCHANGED)

def get_knowledge_prompt_english(str_data):
    """Generate prompt for knowledge point extraction"""
    prompt = '''{text}
1. Think step-by-step through the process of solving this problem.
2. What knowledge points need to be mastered to correctly solve this problem? The granularity of the knowledge points should be individual knowledge entities, equivalent to atoms in a knowledge graph. Provide the name and explanation of each knowledge point.
Respond to the knowledge points in JSON standard format, ensuring that the output can be loaded using json.loads. Do not include any unrelated information or content that is not in JSON format: Keys represent the serial number of the knowledge point, and values are the concatenation of each knowledge point and its corresponding explanation, separated by ' - ':
{"1": "Knowledge Point - Explanation", "2": "Knowledge Point - Explanation", "3": "Knowledge Point - Explanation", ...}
output:'''
    return {"prompt": prompt.replace('{text}', str_data)}

def extract_json_from_result(result):
    """Extract and parse JSON from GPT response"""
    try:
        json_match = re.search(r'```json(.*?)```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_start = result.find('{')
            json_end = result.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = result[json_start:json_end + 1]
            else:
                return None
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

# Unified API client adapter

class UnifiedOpenAIClient:
    """Unified OpenAI client supporting both internal and standard API"""
    
    def __init__(self, api_mode: str = "auto", api_key: str = None, app_key: int = None, model: str = "gpt-4o"):
        """
        Initialize unified OpenAI client
        
        Args:
            api_mode: "internal", "standard", or "auto" (default)
            api_key: OpenAI API key for standard API
            app_key: App key for internal API
            model: Model name
        """
        print(f"[DEBUG] Initializing UnifiedOpenAIClient with mode: {api_mode}")
        
        # 确保API已经导入
        internal_available, standard_available = _lazy_import_apis()
        
        self.model = model
        self.api_mode = self._determine_api_mode(api_mode, api_key, app_key, internal_available, standard_available)
        
        if self.api_mode == "internal":
            if not internal_available:
                raise ValueError("Internal API requested but open_ai_utils not available")
            
            print(f"[DEBUG] Creating internal API client...")
            self.client = InternalOpenAIClient(
                app_key=app_key or "YOUR_INTERNAL_API_KEY_HERE",  # Default app key from reference
                max_tokens=16384,
                timeout=60*1000*5
            )
            
        elif self.api_mode == "standard":
            if not standard_available:
                raise ValueError("Standard API requested but openai library not available")
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required for standard API")
            
            print(f"[DEBUG] Creating standard API client...")
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            print(f"[INFO] Initialized standard API client with model={model}")
            
        else:
            raise ValueError(f"Unable to initialize API client with mode: {api_mode}")
    
    def _determine_api_mode(self, api_mode: str, api_key: str, app_key: int, internal_available: bool, standard_available: bool) -> str:
        """Determine which API mode to use"""
        if api_mode == "internal":
            return "internal"
        elif api_mode == "standard":
            return "standard"
        elif api_mode == "auto":
            # Auto mode: prefer internal API if available
            if internal_available:
                return "internal"
            elif standard_available and (api_key or os.getenv('OPENAI_API_KEY')):
                return "standard"
            else:
                raise ValueError("No suitable API client available in auto mode")
        else:
            raise ValueError(f"Invalid api_mode: {api_mode}")
    
    def n_threads_do(self, n_threads: int, prompts: List[Dict]) -> List[str]:
        """Unified interface for batch processing"""
        if self.api_mode == "internal":
            # Use internal API's native multi-threading
            return self.client.n_threads_do(n_threads, prompts)
        
        elif self.api_mode == "standard":
            # Use standard API with sequential processing
            # Note: This is sequential, not truly multi-threaded
            results = []
            for prompt in tqdm(prompts, desc="API calls"):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt["prompt"]}],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    results.append(response.choices[0].message.content)
                except Exception as e:
                    print(f"[ERROR] API call failed: {e}")
                    results.append(None)
                    
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            return results

# Core processing functions (UNCHANGED)

def process_batch(batch_doc_ids, batch_questions, batch_prompts, model, f_raw, f_parsed, f_error, pbar, examples, max_retry=6, sleep_time=1):
    """Process a batch of requests with retry mechanism"""
    # Retry mechanism for failed requests
    remaining = list(range(len(batch_doc_ids)))
    results = [None] * len(batch_doc_ids)
    raw_outputs = [None] * len(batch_doc_ids)
    
    for attempt in range(1, max_retry + 1):
        if not remaining:
            break
        prompts_to_send = [batch_prompts[i] for i in remaining]
        try:
            batch_results = model.n_threads_do(len(prompts_to_send), prompts_to_send)
        except Exception as e:
            print(f"[ERROR] Batch request failed, attempt {attempt}/{max_retry}, error={e}")
            time.sleep(sleep_time)
            continue
        
        for idx_in_batch, res in enumerate(batch_results):
            i = remaining[idx_in_batch]
            raw_outputs[i] = res
            parsed = extract_json_from_result(res) if res else None
            if parsed is not None:
                results[i] = parsed
        
        remaining = [i for i in remaining if results[i] is None]
        if remaining:
            print(f"[WARN] {len(remaining)} items failed after attempt {attempt}, retrying")
            time.sleep(sleep_time)
    
    # Write outputs
    for i in range(len(batch_doc_ids)):
        doc_id = batch_doc_ids[i]
        question = batch_questions[i]
        gpt4o_output = raw_outputs[i]
        
        # Write raw output
        raw_entry = {"doc_id": doc_id, "prompt": batch_prompts[i], "gpt4o_output": gpt4o_output}
        f_raw.write(json.dumps(raw_entry, ensure_ascii=False) + '\n')
        
        # Write parsed result or error
        if results[i] is not None:
            parsed_entry = {"doc_id": doc_id, "knowledge_points": results[i]}
            f_parsed.write(json.dumps(parsed_entry, ensure_ascii=False) + '\n')
            if len(examples) < 3:
                examples.append(parsed_entry)
            # Count success
            try:
                process_batch.success_count += 1
            except AttributeError:
                process_batch.success_count = 1
        else:
            error_entry = {"doc_id": doc_id, "question": question, "gpt4o_output": gpt4o_output}
            f_error.write(json.dumps(error_entry, ensure_ascii=False) + '\n')
            try:
                process_batch.fail_count += 1
            except AttributeError:
                process_batch.fail_count = 1
        
        pbar.update(1)

# Initialize counters
process_batch.success_count = 0
process_batch.fail_count = 0

def process_file(input_file, dataset_name, output_base_dir, api_mode="auto", api_key=None, app_key=None, model="gpt-4o", max_lines=None, batch_size=100, max_retry=6):
    """Process a single file for knowledge point extraction"""
    
    # Create output directory
    dataset_output_dir = os.path.join(output_base_dir, f"{dataset_name}_gpt4o_output_knowledge_points")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    basename = os.path.basename(input_file)
    prefix = os.path.splitext(basename)[0]
    
    raw_output_file = os.path.join(dataset_output_dir, prefix + "_gpt4o_raw_output.jsonl")
    parsed_output_file = os.path.join(dataset_output_dir, prefix + "_knowledge_points.jsonl")
    error_output_file = os.path.join(dataset_output_dir, prefix + "_error.jsonl")
    
    # Resume from checkpoint: check processed doc_ids
    processed_doc_id = set()
    if os.path.exists(parsed_output_file):
        with open(parsed_output_file, "r", encoding="utf-8") as f_parsed:
            for line in f_parsed:
                try:
                    parsed_data = json.loads(line)
                    processed_doc_id.add(parsed_data['doc_id'])
                except Exception:
                    continue
    
    # Count total lines
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    if max_lines is not None:
        total_lines = min(total_lines, max_lines)
    
    examples = []
    
    # Create unified model client
    model_client = UnifiedOpenAIClient(
        api_mode=api_mode,
        api_key=api_key,
        app_key=app_key,
        model=model
    )
    
    # Reset counters
    process_batch.success_count = 0
    process_batch.fail_count = 0
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(raw_output_file, "a", encoding="utf-8") as f_raw, \
         open(parsed_output_file, "a", encoding="utf-8") as f_parsed, \
         open(error_output_file, "a", encoding="utf-8") as f_error, \
         tqdm(total=total_lines, desc=f"Processing {basename}") as pbar:
        
        batch_doc_ids, batch_questions, batch_prompts = [], [], []
        idx = 0
        
        for line in f_in:
            if max_lines is not None and idx >= max_lines:
                break
            idx += 1
            line = line.strip()
            if not line:
                continue
            
            try:
                doc_id, question = line.split('##@@B', 1)
                doc_id = doc_id.strip()
                question = question.strip()
            except Exception as e:
                print(f"[WARN] Skip malformed line: {e} | {line}")
                pbar.update(1)
                continue
            
            if doc_id in processed_doc_id:
                pbar.update(1)
                continue
            
            prompt = get_knowledge_prompt_english(question)
            batch_doc_ids.append(doc_id)
            batch_questions.append(question)
            batch_prompts.append(prompt)
            
            # Process when batch is full
            if len(batch_doc_ids) == batch_size:
                process_batch(batch_doc_ids, batch_questions, batch_prompts, model_client, 
                            f_raw, f_parsed, f_error, pbar, examples, max_retry)
                batch_doc_ids, batch_questions, batch_prompts = [], [], []
        
        # Process remaining items
        if batch_doc_ids:
            process_batch(batch_doc_ids, batch_questions, batch_prompts, model_client, 
                        f_raw, f_parsed, f_error, pbar, examples, max_retry)
    
    # Print statistics
    total_processed = process_batch.success_count + process_batch.fail_count
    print(f"[STAT] Dataset: {dataset_name}")
    print(f"[STAT] Input file: {input_file}")
    print(f"[STAT] Raw output: {raw_output_file}")
    print(f"[STAT] Parsed output: {parsed_output_file}")
    print(f"[STAT] Error output: {error_output_file}")
    print(f"[STAT] Total samples: {total_lines}")
    print(f"[STAT] Successfully parsed: {process_batch.success_count}")
    print(f"[STAT] Failed (after {max_retry} retries): {process_batch.fail_count}")
    
    if total_processed == 0:
        print("[STAT] Success rate: N/A (no data processed)")
    else:
        print(f"[STAT] Success rate: {(process_batch.success_count/total_processed)*100:.2f}%")
    
    print("[STAT] Sample successful results:")
    for idx, s in enumerate(examples):
        print(f"--- Example {idx+1} ---\n{json.dumps(s, ensure_ascii=False, indent=2)}")
    print("-" * 50)
    
    return {
        'success_count': process_batch.success_count,
        'fail_count': process_batch.fail_count,
        'total_processed': total_processed,
        'output_files': {
            'raw': raw_output_file,
            'parsed': parsed_output_file,
            'error': error_output_file
        }
    }

# Dataset discovery and batch processing functions

def extract_dataset_name(filename: str) -> str:
    """Extract dataset name from filename"""
    if isinstance(filename, Path):
        filename = filename.name
    
    # Remove common suffixes
    name = filename
    suffixes_to_remove = [
        '_with_docid_for_extraction.txt',
        '_for_extraction.txt', 
        '_with_docid.txt',
        '.txt'
    ]
    
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    return name

def discover_datasets(input_dir: str, include: Optional[List[str]] = None, 
                     exclude: Optional[List[str]] = None, 
                     pattern: str = "*.txt") -> List[Tuple[str, str]]:
    """Discover dataset files in input directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return []
    
    files = list(input_path.glob(pattern))
    if not files:
        print(f"[WARNING] No files matching pattern '{pattern}' found in {input_dir}")
        return []
    
    datasets = []
    for file_path in files:
        dataset_name = extract_dataset_name(file_path.name)
        
        if include and dataset_name not in include:
            continue
        if exclude and dataset_name in exclude:
            continue
            
        datasets.append((dataset_name, str(file_path)))
    
    return datasets

def extract_knowledge_points_batch(config: Dict) -> Dict:
    """Batch process multiple datasets for knowledge point extraction"""
    
    # Extract configuration
    input_dir = config.get('input_dir', 'data/processed/for_extraction')
    output_dir = config.get('output_dir', 'data/processed/knowledge_points')
    include_datasets = config.get('include_datasets', None)
    exclude_datasets = config.get('exclude_datasets', None)
    file_pattern = config.get('file_pattern', '*.txt')
    
    # API configuration
    api_mode = config.get('api_mode', 'auto')  # Default to auto (prefer internal)
    api_key = config.get('api_key', os.getenv('OPENAI_API_KEY'))
    app_key = config.get('app_key', None)  # Will use default if None
    model = config.get('model', 'gpt-4o')
    batch_size = config.get('batch_size', 100)
    max_retry = config.get('max_retry', 6)
    max_lines = config.get('max_lines', None)
    
    # Validate API configuration
    if api_mode == "standard" and not api_key:
        raise ValueError("API key required for standard API mode. Please set OPENAI_API_KEY or provide api_key in config.")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover datasets
    datasets = discover_datasets(input_dir, include_datasets, exclude_datasets, file_pattern)
    
    if not datasets:
        print("[ERROR] No datasets found to process")
        return {
            'total_datasets': 0,
            'successful_datasets': 0,
            'failed_datasets': 0,
            'results': []
        }
    
    print(f"[INFO] Found {len(datasets)} datasets to process:")
    for dataset_name, file_path in datasets:
        file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        print(f"  - {dataset_name}: {line_count} lines, {file_size:.1f}MB")
    
    # Process each dataset
    results = []
    successful_count = 0
    failed_count = 0
    
    for dataset_name, file_path in datasets:
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"File: {file_path}")
        print('='*60)
        
        try:
            result = process_file(
                input_file=file_path,
                dataset_name=dataset_name,
                output_base_dir=output_dir,
                api_mode=api_mode,
                api_key=api_key,
                app_key=app_key,
                model=model,
                max_lines=max_lines,
                batch_size=batch_size,
                max_retry=max_retry
            )
            
            result['dataset_name'] = dataset_name
            result['file_path'] = file_path
            results.append(result)
            successful_count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            
            failed_result = {
                'dataset_name': dataset_name,
                'file_path': file_path,
                'success_count': 0,
                'fail_count': 0,
                'total_processed': 0,
                'error': str(e)
            }
            results.append(failed_result)
            failed_count += 1
    
    # Summary
    total_success = sum(r.get('success_count', 0) for r in results)
    total_processed = sum(r.get('total_processed', 0) for r in results)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print('='*60)
    print(f"Total datasets: {len(datasets)}")
    print(f"Successful datasets: {successful_count}")
    print(f"Failed datasets: {failed_count}")
    print(f"Total samples processed: {total_processed}")
    print(f"Total successful extractions: {total_success}")
    if total_processed > 0:
        print(f"Overall success rate: {(total_success/total_processed)*100:.2f}%")
    
    print(f"\nPer-dataset results:")
    for result in results:
        name = result['dataset_name']
        success = result.get('success_count', 0)
        total = result.get('total_processed', 0)
        if 'error' in result:
            print(f"  {name}: ERROR - {result['error']}")
        elif total > 0:
            rate = (success/total)*100
            print(f"  {name}: {success}/{total} ({rate:.1f}%)")
        else:
            print(f"  {name}: No data processed")
    
    return {
        'total_datasets': len(datasets),
        'successful_datasets': successful_count,
        'failed_datasets': failed_count,
        'total_samples_processed': total_processed,
        'total_successful_extractions': total_success,
        'results': results
    }

def extract_small_datasets(api_mode: str = "auto", api_key: str = None, app_key: int = None, input_dir: str = None) -> Dict:
    """Convenience function to extract knowledge points from small datasets"""
    print("[DEBUG] extract_small_datasets called")
    
    config = {
        'input_dir': input_dir or 'data/processed/for_extraction',
        'output_dir': 'data/processed/knowledge_points',
        'exclude_datasets': ['GSM8K_train', 'GSM8K_test'],  # Exclude large datasets
        'api_mode': api_mode,
        'api_key': api_key,
        'app_key': app_key,
        'model': 'gpt-4o',
        'batch_size': 5,  # Smaller batches for stability
        'max_retry': 6
    }
    
    return extract_knowledge_points_batch(config)

# Legacy interface functions (for backward compatibility)

def extract_knowledge_points_from_file(file_path: str, config: Dict) -> Dict:
    """Legacy interface - process single file"""
    
    # Extract dataset name from file path
    dataset_name = config.get('dataset_name', extract_dataset_name(file_path))
    
    # Read parameters from config
    output_dir = config.get('output_dir', 'data/processed/knowledge_points')
    api_mode = config.get('api_mode', 'auto')
    api_key = config.get('api_key', os.getenv('OPENAI_API_KEY'))
    app_key = config.get('app_key', None)
    model = config.get('model', 'gpt-4o')
    max_lines = config.get('max_lines', None)
    batch_size = config.get('batch_size', 100)
    max_retry = config.get('max_retry', 6)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Call single file processing function
    return process_file(
        input_file=file_path,
        dataset_name=dataset_name,
        output_base_dir=output_dir,
        api_mode=api_mode,
        api_key=api_key,
        app_key=app_key,
        model=model,
        max_lines=max_lines,
        batch_size=batch_size,
        max_retry=max_retry
    )

def extract_from_file(file_path: str, config: Dict) -> List[Dict]:
    """Legacy interface - simplified interface returning knowledge points list"""
    
    result = extract_knowledge_points_from_file(file_path, config)
    
    # Read parsed results file
    parsed_file = result['output_files']['parsed']
    knowledge_points = []
    
    if os.path.exists(parsed_file):
        with open(parsed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        doc_id = data['doc_id']
                        kps = data['knowledge_points']
                        
                        # Convert to list format
                        for kp_id, kp_text in kps.items():
                            knowledge_points.append({
                                'doc_id': doc_id,
                                'knowledge_point_id': kp_id,
                                'knowledge_point': kp_text
                            })
                    except Exception as e:
                        print(f"[WARN] Error parsing knowledge points: {e}")
                        continue
    
    return knowledge_points

# 主函数部分
def main():
    """Main function to run knowledge extraction"""
    
    print("CogAtom Knowledge Point Extraction")
    print("="*60)
    
    # 检查输入文件
    input_dir = Path("data/processed/for_extraction")
    if not input_dir.exists():
        print("Error: Input directory not found")
        print(f"Expected: {input_dir}")
        return False
    
    expected_files = [
        "AIME_2024_with_docid_for_extraction.txt",
        "MATH-500_test_with_docid_for_extraction.txt"
    ]
    
    missing_files = []
    for filename in expected_files:
        if not (input_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print("Error: Missing input files:")
        for filename in missing_files:
            print(f"  - {filename}")
        return False
    
    print("Input files found:")
    for filename in expected_files:
        file_path = input_dir / filename
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        print(f"  - {filename}: {line_count} lines")
    
    try:
        print("Starting knowledge extraction...")
        result = extract_small_datasets(api_mode="internal")
        
        print(f"Extraction Complete!")
        print(f"Successful datasets: {result['successful_datasets']}/{result['total_datasets']}")
        print(f"Total samples processed: {result['total_samples_processed']}")
        print(f"Total knowledge points extracted: {result['total_successful_extractions']}")
        
        if result['total_samples_processed'] > 0:
            success_rate = (result['total_successful_extractions'] / result['total_samples_processed']) * 100
            print(f"Overall success rate: {success_rate:.1f}%")
        
        print(f" Output directory: data/processed/knowledge_points/")
        return True
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    # Check if help is requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python knowledge_extraction.py")
        print("Extracts knowledge points from AIME_2024 and MATH-500_test datasets")
        sys.exit(0)
    
    success = main()
    if success:
        print("Knowledge point extraction completed successfully!")
    else:
        print("Knowledge point extraction failed!")
        sys.exit(1)
