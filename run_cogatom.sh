#!/bin/bash
# CogAtom Cognitive Atom Generation Pipeline
# Complete pipeline from raw data to structured prompts

set -e

# Configuration
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_FILE="${PROJECT_ROOT}/pipeline.log"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_stage() {
    echo "================================" | tee -a "$LOG_FILE"
    log "STAGE $1: $2"
    echo "================================" | tee -a "$LOG_FILE"
}

# Check if file exists
check_file() {
    if [[ ! -f "$1" ]]; then
        log "ERROR: File not found: $1"
        exit 1
    fi
}

# Check if directory exists
check_dir() {
    if [[ ! -d "$1" ]]; then
        log "ERROR: Directory not found: $1"
        exit 1
    fi
}

# Verify stage output
verify_output() {
    if [[ -f "$1" ]] || [[ -d "$1" ]]; then
        log "SUCCESS: Output verified: $1"
    else
        log "ERROR: Expected output not found: $1"
        exit 1
    fi
}

# Main pipeline
main() {
    cd "$PROJECT_ROOT"
    
    log "Starting CogAtom pipeline..."
    log "Project root: $PROJECT_ROOT"
    
    # Stage 1: Data preprocessing
    log_stage "1" "Data Preprocessing"
    check_dir "data/raw/download_data"
    python scripts/process_data.py
    verify_output "data/processed/for_extraction"
    
    # Stage 2: Knowledge extraction
    log_stage "2" "Knowledge Extraction"
    python knowledge_extraction.py
    verify_output "data/processed/knowledge_points"
    
    # Stage 3: Knowledge consolidation
    log_stage "3" "Knowledge Consolidation"
    python knowledge_consolidation.py
    verify_output "data/processed/knowledge_consolidation/consolidated_knowledge_points.txt"
    
    # Stage 4: Embedding generation
    log_stage "4" "Embedding Generation"
    cd embeddings/
    python extract_embeddings.py
    cd ..
    verify_output "data/processed/embeddings"
    
    # Stage 5: Clustering analysis
    log_stage "5" "Clustering Analysis"
    python cluster/knowledge_point_clustering.py
    verify_output "data/processed/clustering"
    
    # Stage 6: Knowledge list building
    log_stage "6" "Knowledge List Building"
    python save_knowledge_list.py
    verify_output "data/processed/knowledge_points/processed/cogatom_knowledge_list.json"
    
    # Stage 7: Cooccurrence analysis
    log_stage "7" "Cooccurrence Analysis"
    python cooccurrence_analysis/cogatom_cooccurrence_analysis.py
    verify_output "data/processed/graphs/cooccurrence"
    
    # Stage 8: Graph construction
    log_stage "8" "Graph Construction"
    python graph_construction/cogatom_graph_constructor.py
    verify_output "data/processed/knowledge_graphs"
    
    # Stage 9: Random walk (manual step)
    log_stage "9" "Random Walk Paths"
    if [[ ! -f "data/processed/random_walks/cogatom_247_90_diverse/paths/diverse_random_walk_paths.txt" ]]; then
        log "WARNING: Random walk paths not found. Please run random walk script manually."
        log "Expected output: data/processed/random_walks/cogatom_247_90_diverse/paths/diverse_random_walk_paths.txt"
        exit 1
    fi
    log "Random walk paths found, continuing..."
    
    # Stage 10: Dependency extraction
    log_stage "10" "Dependency Extraction"
    python cogatom_dependency_extractor.py
    verify_output "data/processed/dependencies/cogatom_247_90_diverse"
    
    # Stage 11: Cognitive atom generation
    log_stage "11" "Cognitive Atom Generation"
    python cognitive_atom_generator.py
    verify_output "data/processed/combinations/cogatom_247_90_diverse/cognitive_atom_combinations.txt"
    
    # Stage 12: Prompt generation
    log_stage "12" "Prompt Generation"
    python cognitive_atom_prompt_generator.py
    verify_output "data/processed/prompts/cogatom_247_90_diverse/cognitive_atom_synthesis_prompts.jsonl"
    
    # Final summary
    echo "================================" | tee -a "$LOG_FILE"
    log "CogAtom pipeline completed successfully!"
    log "Final output: data/processed/prompts/cogatom_247_90_diverse/cognitive_atom_synthesis_prompts.jsonl"
    
    # Output statistics
    if [[ -f "data/processed/prompts/cogatom_247_90_diverse/cognitive_atom_synthesis_prompts.jsonl" ]]; then
        local prompt_count=$(wc -l < "data/processed/prompts/cogatom_247_90_diverse/cognitive_atom_synthesis_prompts.jsonl")
        log "Generated $prompt_count structured prompts"
    fi
    
    log "Pipeline log saved to: $LOG_FILE"
    echo "================================" | tee -a "$LOG_FILE"
}

# Run pipeline
main "$@"
