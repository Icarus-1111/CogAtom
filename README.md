# CogAtom: Cognitive Atom Generation Pipeline

A complete pipeline for generating cognitive atom combinations from mathematical problem datasets using knowledge extraction, dependency modeling, and intelligent combination generation.

## Quick Start

### Prerequisites

- Python >= 3.8
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```
- Raw datasets downloaded to `data/raw/download_data/` (AIME, MATH-500,......)
- API configuration for GPT-4o calls
- BGE-M3 model downloaded locally

### Option 1: Complete Pipeline (Recommended)

Run the entire pipeline with a single command:

```bash
chmod +x run_cogatom.sh
./run_cogatom.sh
```

### Option 2: Step-by-Step Execution

Execute individual stages for debugging or customization:

```bash
# Stage 1: Data preprocessing
python scripts/process_data.py

# Stage 2: Knowledge extraction
python knowledge_extraction.py

# Stage 3: Knowledge consolidation
python knowledge_consolidation.py

# Stage 4: Embedding generation
cd embeddings/
python extract_embeddings.py --model_path /path/to/your/bge-m3
cd ..

# Stage 5: Clustering analysis
python cluster/knowledge_point_clustering.py

# Stage 6: Knowledge list building
python save_knowledge_list.py

# Stage 7: Cooccurrence analysis
python cooccurrence_analysis/cogatom_cooccurrence_analysis.py

# Stage 8: Graph construction
python graph_construction/cogatom_graph_constructor.py

# Stage 9: Random walk
# Generate paths to: data/processed/random_walks/cogatom_247_90_diverse/paths/diverse_random_walk_paths.txt

# Stage 10: Dependency extraction
python cogatom_dependency_extractor.py

# Stage 11: Cognitive atom generation
python cognitive_atom_generator.py

# Stage 12: Prompt generation
python cognitive_atom_prompt_generator.py
```

## Configuration

### API Settings
Configure API mode in `knowledge_extraction.py` and `cogatom_dependency_extractor.py`:
- `api_mode = "internal"` - Use internal API with app_key
- `api_mode = "standard"` - Use OpenAI API with OPENAI_API_KEY environment variable
- `api_mode = "auto"` - Automatically select available API

### Model Setup
Download BGE-M3 model and specify the path when running embedding generation:
```bash
python extract_embeddings.py --model_path /path/to/your/bge-m3
```

### Algorithm Parameters
Modify parameters in `cognitive_atom_generator.py`:
```python
class CogAtomConfig:
    COMBO_SIZE = 10                    # Target combination size
    N_STARTS = 3                       # Multi-start extension points
    MAX_STEPS = 4                      # Maximum extension steps
    SIMILARITY_THRESHOLD = 0.9         # Clustering threshold
```

## Output

The pipeline generates:
- **Final output**: `data/processed/prompts/cogatom_247_90_diverse/cognitive_atom_synthesis_prompts.jsonl`
- **Intermediate results**: Knowledge points, embeddings, clusters, graphs, dependencies, combinations
- **Logs**: `pipeline.log` for complete execution tracking

## Troubleshooting

- **API failures**: Check network connection and API keys
- **Memory issues**: Reduce batch sizes in respective scripts
- **Missing files**: Ensure all input files exist before running
- **Permission errors**: Check read/write permissions for data directories
- **Model path errors**: Ensure BGE-M3 model path is correct

For detailed progress monitoring:
```bash
tail -f pipeline.log
```
