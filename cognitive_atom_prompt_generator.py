#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

class PromptTemplateManager:
    """Manages prompt templates for different synthesis methods."""
    
    def __init__(self):
        """Initialize with predefined synthesis templates."""
        self.templates = {
            "synthesize_and_expand_problem_no_answer": self._get_synthesis_template()
        }
    
    def _get_synthesis_template(self) -> str:
        """Returns the main synthesis template for problem generation."""
        return '''What advanced mathematics or mathematics competition problems are related to [{key}]? Please follow the steps below to design a new comprehensive advanced mathematics or mathematics competition problem. You do NOT need to provide the answer or solution process.

Steps:
First select the main knowledge points that are most suitable for combining into a problem (try to minimize discarding knowledge points at this stage, but you may omit a few that are clearly unsuitable for integration). Design an initial version of the problem that integrates these main knowledge points.

Next, to increase the diversity and difficulty of the problem, please actively consider which advanced knowledge points (not limited to the original set) can be reasonably and naturally integrated into the problem, based on the structure and objectives of the initial problem. Supplement your initial selection with these higher-level knowledge points that are more challenging and easier to integrate. Based on the initial problem, design a new, more difficult and diverse advanced calculus problem by integrating these additional knowledge points.

Clearly specify:
The initial version of the problem (with the main knowledge points used for the first combination)
The knowledge points discarded (relative to the original set provided)
The knowledge points added (including those beyond the original set, if any)
The design rationale for the more difficult, extended version of the problem, explaining why you chose these additional knowledge points and how they improve the problem's difficulty, comprehensiveness, and educational value
The final, extended version of the problem

Output format requirement: Use standard JSON format, strictly following the structure below, and ensure the result can be loaded via json.loads. Do not include any irrelevant information or non-JSON content:
```json
{"initial_problem": xxx,"discarded_knowledge_points": xxx,"added_knowledge_points": xxx,"extended_design_idea": xxx,"extended_problem": xxx}```'''
    
    def get_template(self, template_name: str) -> Optional[str]:
        """Get template by name."""
        return self.templates.get(template_name)
    
    def get_all_templates(self) -> Dict[str, str]:
        """Get all available templates."""
        return self.templates.copy()

class KnowledgePointMapper:
    """Handles mapping of knowledge points to their descriptions."""
    
    def __init__(self, node_dict_file: str):
        """
        Initialize with knowledge point dictionary file.
        
        Args:
            node_dict_file: Path to JSON file containing knowledge point descriptions
        """
        self.node_dict = self._load_node_dict(node_dict_file)
        self.missing_count = 0
    
    def _load_node_dict(self, node_dict_file: str) -> Dict[str, str]:
        """Load knowledge point dictionary from JSON file."""
        try:
            with open(node_dict_file, 'r', encoding='utf-8') as f:
                node_dict = json.load(f)
            print(f"[INFO] Knowledge point dictionary loaded: {len(node_dict)} entries")
            return node_dict
        except Exception as e:
            print(f"[ERROR] Failed to load node dictionary: {e}")
            return {}
    
    def _clean_key(self, key: str) -> str:
        """Clean knowledge point key by removing escape characters and quotes."""
        return key.replace("\\", "").strip().strip("'\"")
    
    def _fuzzy_find_key(self, key: str) -> Tuple[bool, str]:
        """
        Perform fuzzy search for knowledge point key.
        
        Args:
            key: Knowledge point key to search
            
        Returns:
            Tuple of (found, matched_key)
        """
        # Try exact match first
        if key in self.node_dict:
            return True, key
        
        # Try with stripped quotes
        stripped_key = key.strip("'\"")
        if stripped_key in self.node_dict:
            return True, stripped_key
        
        return False, key
    
    def map_knowledge_points(self, knowledge_points: List[str]) -> Tuple[List[str], List[str]]:
        """
        Map knowledge points to their descriptions.
        
        Args:
            knowledge_points: List of knowledge point keys
            
        Returns:
            Tuple of (found_explanations, missing_keys)
        """
        found_explanations = []
        missing_keys = []
        
        for kp in knowledge_points:
            cleaned_key = self._clean_key(kp)
            found, matched_key = self._fuzzy_find_key(cleaned_key)
            
            if found:
                explanation = f"{matched_key} - {self.node_dict[matched_key]}"
                found_explanations.append(explanation)
            else:
                missing_keys.append(cleaned_key)
                self.missing_count += 1
        
        return found_explanations, missing_keys

class PromptGenerator:
    """Main class for generating prompts from cognitive atom combinations."""
    
    def __init__(self, config_paths: Dict[str, str]):
        """
        Initialize the prompt generator.
        
        Args:
            config_paths: Dictionary containing file paths configuration
        """
        self.config = config_paths
        self.template_manager = PromptTemplateManager()
        self.knowledge_mapper = KnowledgePointMapper(config_paths['node_dict_file'])
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_outputs': 0,
            'json_errors': 0,
            'missing_knowledge_points': 0
        }
        
        # Setup output paths
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_jsonl = self.output_dir / "cognitive_atom_synthesis_prompts.jsonl"
        self.failure_log = self.output_dir / "missing_knowledge_points.jsonl"
    
    def _parse_combination_line(self, line: str) -> List[str]:
        """Parse a single line containing knowledge point combination."""
        return [kp.strip() for kp in line.split('##') if kp.strip()]
    
    def _generate_prompt_from_template(self, template: str, key_string: str) -> str:
        """Generate prompt by substituting key into template."""
        return template.replace('{key}', key_string)
    
    def _create_output_object(self, prompt: str, method_name: str, 
                             doc_id: str, combine_method: str) -> Dict[str, str]:
        """Create structured output object for JSONL."""
        return {
            "deepseek": prompt,
            "synthesis_method": method_name,
            "doc_id": doc_id,
            "combine_method": combine_method
        }
    
    def _log_missing_knowledge_points(self, line_no: int, input_file: str,
                                    combine_method: str, raw_line: str,
                                    knowledge_points: List[str], 
                                    missing_points: List[str]) -> None:
        """Log combinations with missing knowledge point descriptions."""
        failure_obj = {
            "line_number": line_no,
            "source_file": os.path.basename(input_file),
            "combination_method": combine_method,
            "raw_combination": raw_line,
            "knowledge_points": knowledge_points,
            "missing_knowledge_points": missing_points,
            "missing_count": len(missing_points)
        }
        
        with open(self.failure_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(failure_obj, ensure_ascii=False) + '\n')
    
    def process_combination_file(self, input_file: str, combine_method: str, 
                               append_mode: bool = False) -> Tuple[int, int, int, int]:
        """
        Process a single combination file and generate prompts.
        
        Args:
            input_file: Path to input combination file
            combine_method: Name of the combination method used
            append_mode: Whether to append to existing output file
            
        Returns:
            Tuple of (total_lines, successful_outputs, json_errors, missing_kp_count)
        """
        print(f"[INFO] Processing file: {os.path.basename(input_file)}")
        print(f"[INFO] Combination method: {combine_method}")
        
        # Read input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"[ERROR] Failed to read input file: {e}")
            return 0, 0, 0, 0
        
        print(f"[INFO] Total combinations in file: {len(lines)}")
        
        # Process combinations
        file_stats = {'total': 0, 'success': 0, 'errors': 0, 'missing': 0}
        mode = 'a' if append_mode else 'w'
        
        with open(self.output_jsonl, mode, encoding='utf-8') as output_file:
            for idx, line in enumerate(tqdm(lines, desc=f"Generating prompts ({combine_method})")):
                file_stats['total'] += 1
                
                # Parse knowledge points
                knowledge_points = self._parse_combination_line(line)
                if not knowledge_points:
                    continue
                
                # Map to descriptions
                explanations, missing_keys = self.knowledge_mapper.map_knowledge_points(knowledge_points)
                
                # Skip if any knowledge points are missing
                if missing_keys:
                    file_stats['missing'] += 1
                    self._log_missing_knowledge_points(
                        idx + 1, input_file, combine_method, line,
                        knowledge_points, missing_keys
                    )
                    continue
                
                # Generate prompts for all templates
                key_string = '] and ['.join(explanations)
                doc_id = f"{os.path.basename(input_file)}_line_{idx+1:07d}"
                
                for template_name, template in self.template_manager.get_all_templates().items():
                    if not template.strip():
                        continue
                    
                    try:
                        prompt = self._generate_prompt_from_template(template, key_string)
                        output_obj = self._create_output_object(
                            prompt, template_name, doc_id, combine_method
                        )
                        
                        json_line = json.dumps(output_obj, ensure_ascii=False)
                        output_file.write(json_line + '\n')
                        file_stats['success'] += 1
                        
                    except Exception as e:
                        print(f"[ERROR] JSON generation failed for line {idx+1}, method {template_name}: {e}")
                        file_stats['errors'] += 1
        
        print(f"[INFO] File processing completed for {combine_method}")
        print(f"[INFO] Successful outputs: {file_stats['success']}, "
              f"Errors: {file_stats['errors']}, Missing KP: {file_stats['missing']}")
        
        return file_stats['total'], file_stats['success'], file_stats['errors'], file_stats['missing']
    
    def validate_jsonl_output(self) -> None:
        """Validate that all lines in output JSONL are valid JSON."""
        print(f"[INFO] Validating JSONL output: {self.output_jsonl}")
        
        total_lines = 0
        valid_lines = 0
        invalid_lines = 0
        
        try:
            with open(self.output_jsonl, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    total_lines += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        json.loads(line)
                        valid_lines += 1
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Invalid JSON at line {line_no}: {e}")
                        invalid_lines += 1
        
        except Exception as e:
            print(f"[ERROR] Failed to validate output file: {e}")
            return
        
        print(f"[INFO] Validation completed - Total: {total_lines}, "
              f"Valid: {valid_lines}, Invalid: {invalid_lines}")
    
    def run(self, input_files: List[str], combine_methods: List[str]) -> None:
        """
        Run the complete prompt generation pipeline.
        
        Args:
            input_files: List of input combination files
            combine_methods: List of combination method names
        """
        print("CogAtom Cognitive Atom Prompt Generator")
        print("=" * 50)
        
        # Clear failure log
        if self.failure_log.exists():
            self.failure_log.unlink()
        
        # Process each file
        for i, (input_file, combine_method) in enumerate(zip(input_files, combine_methods)):
            if not os.path.exists(input_file):
                print(f"[WARNING] Input file not found: {input_file}")
                continue
            
            append_mode = (i > 0)
            file_total, file_success, file_errors, file_missing = self.process_combination_file(
                input_file, combine_method, append_mode
            )
            
            # Update global statistics
            self.stats['total_processed'] += file_total
            self.stats['successful_outputs'] += file_success
            self.stats['json_errors'] += file_errors
            self.stats['missing_knowledge_points'] += file_missing
        
        # Validate output
        self.validate_jsonl_output()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print processing summary statistics."""
        print("\n" + "=" * 50)
        print("Processing Summary:")
        print(f"  Total combinations processed: {self.stats['total_processed']}")
        print(f"  Successful prompt generations: {self.stats['successful_outputs']}")
        print(f"  JSON generation errors: {self.stats['json_errors']}")
        print(f"  Missing knowledge point descriptions: {self.stats['missing_knowledge_points']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful_outputs'] / self.stats['total_processed']) * 100
            print(f"  Success rate: {success_rate:.2f}%")
        
        print(f"\nOutput files:")
        print(f"  Prompts: {self.output_jsonl}")
        print(f"  Missing KP log: {self.failure_log}")
        print("\nPrompt generation completed successfully!")

def main():
    """Main execution function."""
    
    # Configuration with correct relative paths pointing to existing files
    script_dir = Path(__file__).parent
    
    config_paths = {
        # Use the existing knowledge point dictionary file
        'node_dict_file': str(script_dir / "data" / "processed" / "knowledge_consolidation" / 
                             "knowledge_point_dictionary.json"),
        'output_dir': str(script_dir / "data" / "processed" / "prompts" / "cogatom_247_90_diverse")
    }
    
    # Input files and methods - using the generated combinations
    input_files = [
        str(script_dir / "data" / "processed" / "combinations" / "cogatom_247_90_diverse" / 
            "cognitive_atom_combinations.txt")
    ]
    
    combine_methods = [
        "cognitive_atom_order_independent_dedup"
    ]
    
    # Create and run generator
    generator = PromptGenerator(config_paths)
    generator.run(input_files, combine_methods)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
