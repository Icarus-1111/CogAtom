#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import re
from tqdm import tqdm
from pathlib import Path

# Delayed import flags
INTERNAL_API_AVAILABLE = None
STANDARD_API_AVAILABLE = None
InternalOpenAIClient = None

def _lazy_import_apis():
    """Lazy import API clients to avoid blocking during module loading"""
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
        
        # Ensure APIs are imported
        internal_available, standard_available = _lazy_import_apis()
        
        self.model = model
        self.api_mode = self._determine_api_mode(api_mode, api_key, app_key, internal_available, standard_available)
        
        if self.api_mode == "internal":
            if not internal_available:
                raise ValueError("Internal API requested but open_ai_utils not available")
            
            print(f"[DEBUG] Creating internal API client...")
            self.client = InternalOpenAIClient(
                app_key=app_key or 1791012913151668297,  # Default app key
                max_tokens=16384,
                timeout=60*1000*5
            )
            print(f"[INFO] Initialized internal API client with app_key={app_key or 1791012913151668297}")
            
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
    
    def n_threads_do(self, n_threads: int, prompts: list) -> list:
        """Unified interface for batch processing"""
        if self.api_mode == "internal":
            # Use internal API's native multi-threading
            return self.client.n_threads_do(n_threads, prompts)
        
        elif self.api_mode == "standard":
            # Use standard API with sequential processing
            results = []
            for prompt in prompts:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt["prompt"]}],
                        temperature=0.1,
                        max_tokens=4000
                    )
                    results.append(response.choices[0].message.content)
                except Exception as e:
                    print(f"[ERROR] API call failed: {e}")
                    results.append(None)
                    
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            return results

# ========== Configuration ==========

INPUT_FILE = "./data/processed/random_walks/cogatom_247_90_diverse/paths/diverse_random_walk_paths.txt"
OUTPUT_DIR = "./data/processed/dependencies/cogatom_247_90_diverse"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_RETRY = 6
SLEEP_TIME = 1
BATCH_SIZE = 100
MAX_LINES_TO_PROCESS = None  # None means process all

def get_dependency_prompt_en(knowledge_points):
    """Generate English prompt for dependency extraction"""
    kp_list_str = json.dumps(knowledge_points, ensure_ascii=False)
    prompt = (
        "You are a senior mathematics education expert specializing in curriculum design and knowledge structure analysis.\n\n"
        "Given the following ordered list of mathematical knowledge points, analyze all possible knowledge point pairs (A, B) where A appears before B.\n"
        "Output ONLY those pairs where A is a prerequisite (dependency) for B, with dependency strength scores of 3 (moderate), 4 (strong), or 5 (essential):\n"
        "- 3 (Moderate): A is helpful for B; mastering A makes learning B easier, but B can be understood through alternative approaches.\n"
        "- 4 (Strong): A is very important for B; without understanding A, learning B would be significantly impaired.\n"
        "- 5 (Essential): A is a strict prerequisite for B; without mastering A, one cannot properly understand or learn B.\n"
        "For each dependency relationship, provide the score (3, 4, or 5) and a concise academic justification.\n\n"
        f"Input knowledge points:\n{kp_list_str}\n\n"
        "Output format (JSON only, minimize line breaks):\n"
        '[{"from": "A", "to": "B", "score": 3, "reason": "concise academic justification"}, {"from": "A", "to": "C", "score": 4, "reason": "concise academic justification"}]\n'
        "Do not output pairs with scores 1 or 2. Output only the JSON array above, write all dependency pairs in a single line, avoid extra line breaks."
    )
    return {"prompt": prompt}

def extract_json_from_result(result):
    """Extract JSON from GPT response"""
    try:
        json_start = result.find('[')
        json_end = result.rfind(']')
        if json_start != -1 and json_end != -1:
            json_str = result[json_start:json_end + 1]
            return json.loads(json_str)
        return None
    except Exception:
        return None

def parse_cogatom_walk_line(line):
    """Parse CogAtom diverse random walk line format"""
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
            # Has weight information, take first part
            kp_str = path_part
        else:
            # Check if contains weight information
            if "##@@B" in path_part:
                kp_str = path_part.split("##@@B")[0]
            else:
                kp_str = path_part
        
        # Split and clean nodes
        nodes = [x.strip() for x in kp_str.split("##") if x.strip()]
        
        # Combine start node with path nodes, deduplicate but preserve order
        all_nodes = [start_node] + nodes
        seen = set()
        kp_list = []
        for node in all_nodes:
            if node not in seen:
                seen.add(node)
                kp_list.append(node)
        
        return kp_list if len(kp_list) >= 2 else None
    except Exception:
        return None

def process_file(input_file, max_lines=None):
    """Main file processing function"""
    raw_output_file = os.path.join(OUTPUT_DIR, "diverse_random_walk_gpt4o_raw_output.jsonl")
    parsed_output_file = os.path.join(OUTPUT_DIR, "diverse_random_walk_dependencies.jsonl")
    error_output_file = os.path.join(OUTPUT_DIR, "diverse_random_walk_error.jsonl")

    # Checkpoint resume: check processed lines
    processed_index = set()
    if os.path.exists(parsed_output_file):
        with open(parsed_output_file, "r", encoding="utf-8") as f_parsed:
            for line in f_parsed:
                try:
                    parsed_data = json.loads(line)
                    processed_index.add(parsed_data['line_index'])
                except Exception:
                    continue

    # Count total lines
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    if max_lines is not None:
        total_lines = min(total_lines, max_lines)
    examples = []

    # Create model client
    model = UnifiedOpenAIClient(api_mode="auto", model="gpt-4o")

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(raw_output_file, "a", encoding="utf-8") as f_raw, \
         open(parsed_output_file, "a", encoding="utf-8") as f_parsed, \
         open(error_output_file, "a", encoding="utf-8") as f_error, \
         tqdm(total=total_lines, desc="Processing diverse_random_walk_paths.txt") as pbar:
        
        batch_indices, batch_kp_lists, batch_prompts = [], [], []
        idx = 0
        
        for line in f_in:
            if max_lines is not None and idx >= max_lines:
                break
            
            line = line.strip()
            if not line:
                continue
            
            kp_list = parse_cogatom_walk_line(line)
            if kp_list is None or len(kp_list) < 2:
                pbar.update(1)
                idx += 1
                continue
            
            if idx in processed_index:
                pbar.update(1)
                idx += 1
                continue
            
            prompt = get_dependency_prompt_en(kp_list)
            batch_indices.append(idx)
            batch_kp_lists.append(kp_list)
            batch_prompts.append(prompt)
            
            if len(batch_indices) == BATCH_SIZE:
                process_batch(batch_indices, batch_kp_lists, batch_prompts, model, f_raw, f_parsed, f_error, pbar, examples)
                batch_indices, batch_kp_lists, batch_prompts = [], [], []
            
            idx += 1
        
        # Process remaining batch
        if batch_indices:
            process_batch(batch_indices, batch_kp_lists, batch_prompts, model, f_raw, f_parsed, f_error, pbar, examples)
    
    # Output statistics
    total_processed = process_batch.success_count + process_batch.fail_count
    print(f"[STAT] Input file: {input_file}")
    print(f"[STAT] Raw output: {raw_output_file}")
    print(f"[STAT] Parsed output: {parsed_output_file}")
    print(f"[STAT] Error output: {error_output_file}")
    print(f"[STAT] Total samples: {total_lines}")
    print(f"[STAT] Successfully parsed: {process_batch.success_count}")
    print(f"[STAT] Failed (after {MAX_RETRY} retries): {process_batch.fail_count}")
    if total_processed == 0:
        print("[STAT] Success rate: N/A (no data processed this round)")
    else:
        print(f"[STAT] Success rate: {(process_batch.success_count/total_processed)*100:.2f}%")
    print("[STAT] Sample successful results:")
    for idx, s in enumerate(examples):
        print(f"--- Example {idx+1} ---\n{json.dumps(s, ensure_ascii=False, indent=2)}")
    print("-" * 50)

def process_batch(batch_indices, batch_kp_lists, batch_prompts, model, f_raw, f_parsed, f_error, pbar, examples):
    """Process a batch with retry mechanism"""
    remaining = list(range(len(batch_indices)))
    results = [None] * len(batch_indices)
    raw_outputs = [None] * len(batch_indices)
    
    for attempt in range(1, MAX_RETRY+1):
        if not remaining:
            break
        
        prompts_to_send = [batch_prompts[i] for i in remaining]
        try:
            batch_results = model.n_threads_do(len(prompts_to_send), prompts_to_send)
        except Exception as e:
            print(f"[ERROR] Batch request failed, attempt {attempt}/{MAX_RETRY}, error={e}")
            time.sleep(SLEEP_TIME)
            continue
        
        for idx_in_batch, res in enumerate(batch_results):
            i = remaining[idx_in_batch]
            raw_outputs[i] = res
            parsed = extract_json_from_result(res)
            if parsed is not None:
                results[i] = parsed
        
        remaining = [i for i in remaining if results[i] is None]
        if remaining:
            print(f"[WARN] {len(remaining)} items still unparsed after attempt {attempt}, retrying")
            time.sleep(SLEEP_TIME)
    
    # Save results
    for i in range(len(batch_indices)):
        line_index = batch_indices[i]
        kp_list = batch_kp_lists[i]
        gpt4o_output = raw_outputs[i]
        
        # Save raw output
        raw_entry = {
            "line_index": line_index, 
            "knowledge_points": kp_list, 
            "prompt": batch_prompts[i], 
            "gpt4o_output": gpt4o_output
        }
        f_raw.write(json.dumps(raw_entry, ensure_ascii=False) + '\n')
        
        if results[i] is not None:
            # Successfully parsed
            parsed_entry = {
                "line_index": line_index, 
                "knowledge_points": kp_list, 
                "dependencies": results[i]
            }
            f_parsed.write(json.dumps(parsed_entry, ensure_ascii=False) + '\n')
            if len(examples) < 3:
                examples.append(parsed_entry)
            try:
                process_batch.success_count += 1
            except AttributeError:
                process_batch.success_count = 1
        else:
            # Parse failed
            error_entry = {
                "line_index": line_index, 
                "knowledge_points": kp_list, 
                "gpt4o_output": gpt4o_output
            }
            f_error.write(json.dumps(error_entry, ensure_ascii=False) + '\n')
            try:
                process_batch.fail_count += 1
            except AttributeError:
                process_batch.fail_count = 1
        
        pbar.update(1)

# Initialize counters
process_batch.success_count = 0
process_batch.fail_count = 0

def main():
    """Main execution function"""
    print("="*60)
    print("【CogAtom Knowledge Dependency Extraction with GPT-4o】")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Check input file
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return 1
    
    # Get file info
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    file_size = Path(INPUT_FILE).stat().st_size / 1024 / 1024  # MB
    print(f"Input file info: {line_count} lines, {file_size:.1f}MB")
    
    # Reset counters
    process_batch.success_count = 0
    process_batch.fail_count = 0
    
    # Start processing
    print("\nStarting dependency extraction...")
    start_time = time.time()
    
    try:
        process_file(INPUT_FILE, max_lines=MAX_LINES_TO_PROCESS)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nProcessing completed!")
        print(f"Total time: {duration:.1f} seconds")
        print(f"Output files saved to: {OUTPUT_DIR}")
        
        return 0
        
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
