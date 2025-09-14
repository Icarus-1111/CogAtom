#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import random
import argparse
import multiprocessing
from pathlib import Path
import logging

import torch
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root():
    """Get the project root directory (cogatom/)"""
    # Get the directory where this script is located (embeddings/)
    script_dir = Path(__file__).parent
    # Go up one level to get project root (cogatom/)
    project_root = script_dir.parent
    return project_root


def get_node_rank_and_node_nums():
    """Get node rank and total number of nodes from cluster environment"""
    try:
        cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
        role = cluster_spec["role"]
        assert role == "worker", "{} vs worker".format(role)
        node_rank = cluster_spec["index"]
        nnodes = len(cluster_spec[role])
        return int(node_rank), nnodes
    except (KeyError, json.JSONDecodeError, AssertionError):
        # Single node mode
        return 0, 1


def extract_knowledge_point_names(file_path):
    """
    Extract unique knowledge point names from consolidated knowledge points file
    
    Args:
        file_path: Path to consolidated_knowledge_points.txt
        
    Returns:
        List of unique knowledge point names (sorted)
    """
    unique_names = set()
    total_lines = 0
    valid_lines = 0
    
    if not os.path.exists(file_path):
        logger.error("Input file not found: {}".format(file_path))
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                
                if line:  # Non-empty line
                    valid_lines += 1
                    if ' - ' in line:
                        # Extract knowledge point name (before " - ")
                        name = line.split(' - ')[0].strip()
                        if name:
                            unique_names.add(name)
                    else:
                        # Use entire line as knowledge point name
                        unique_names.add(line)
                        
    except Exception as e:
        logger.error("Error reading file {}: {}".format(file_path, e))
        return []
    
    knowledge_point_names = sorted(list(unique_names))  # Sort for consistent ordering
    
    logger.info("File processing statistics:")
    logger.info("  Total lines: {}".format(total_lines))
    logger.info("  Valid knowledge point lines: {}".format(valid_lines))
    logger.info("  Unique knowledge point names: {}".format(len(knowledge_point_names)))
    
    # Print some examples
    logger.info("Sample knowledge point names:")
    for i, name in enumerate(knowledge_point_names[:10]):
        logger.info("  {}: {}".format(i+1, name))
    if len(knowledge_point_names) > 10:
        logger.info("  ... and {} more".format(len(knowledge_point_names) - 10))
    
    return knowledge_point_names


def bert_encode(model, texts, batch_size=256):
    """
    Encode texts using BGE-M3 model
    
    Args:
        model: BGE-M3 model instance
        texts: List of texts to encode
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of (dense_embeddings, sparse_embeddings)
    """
    dense_vecs = []
    sparse_vecs = []
    
    logger.info("Encoding {} knowledge point names with batch size {}".format(len(texts), batch_size))
    
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i:i+batch_size]
            
            # Encode batch
            embeddings = model.encode(
                batch_texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            dense_vecs.append(embeddings['dense_vecs'])
            if 'lexical_weights' in embeddings:
                sparse_vecs.extend(embeddings['lexical_weights'])
            
    except Exception as e:
        logger.error("Error during encoding: {}".format(e))
        raise
    
    # Concatenate dense embeddings
    dense_embeddings = np.vstack(dense_vecs) if dense_vecs else np.array([])
    
    logger.info("Encoding complete. Dense shape: {}".format(dense_embeddings.shape))
    
    return dense_embeddings, sparse_vecs


def save_embeddings_and_metadata(
    dense_embeddings,
    sparse_embeddings,
    knowledge_point_names,
    output_dir,
    model_path,
    processing_time
):
    """
    Save embeddings and associated metadata
    
    Args:
        dense_embeddings: Dense embedding matrix
        sparse_embeddings: Sparse embedding data
        knowledge_point_names: List of knowledge point names
        output_dir: Output directory
        model_path: Path to the embedding model
        processing_time: Total processing time
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save dense embeddings
        dense_file = output_path / "knowledge_point_names_dense.npy"
        np.save(dense_file, dense_embeddings)
        logger.info("Saved dense embeddings to {}".format(dense_file))
        
        # Save sparse embeddings
        sparse_file = output_path / "knowledge_point_names_sparse.npy"
        np.save(sparse_file, sparse_embeddings, allow_pickle=True)
        logger.info("Saved sparse embeddings to {}".format(sparse_file))
        
        # Save knowledge point names list
        list_file = output_path / "knowledge_point_names_list.txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            for i, name in enumerate(knowledge_point_names):
                f.write("{}\t{}\n".format(i, name))  # Include index for easy reference
        logger.info("Saved knowledge point names list to {}".format(list_file))
        
        # Save metadata
        metadata = {
            "model_path": model_path,
            "num_knowledge_point_names": len(knowledge_point_names),
            "dense_embedding_shape": list(dense_embeddings.shape),
            "sparse_embedding_count": len(sparse_embeddings),
            "processing_time_seconds": processing_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_config": {
                "use_fp16": True,
                "pooling_method": "cls",
                "return_dense": True,
                "return_sparse": True
            },
            "data_info": {
                "clustering_target": "knowledge_point_names",
                "extraction_method": "name_before_dash",
                "sorted": True
            }
        }
        
        metadata_file = output_path / "embedding_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("Saved metadata to {}".format(metadata_file))
        
        # Save processing log
        log_file = output_path / "processing_log.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Knowledge Point Names Embedding Extraction Log\n")
            f.write("=" * 55 + "\n\n")
            f.write("Purpose: Extract embeddings for knowledge point names for clustering\n")
            f.write("Target: Knowledge point names (text before ' - ' separator)\n\n")
            f.write("Input file: {}\n".format(getattr(args, 'input_file', 'N/A') if 'args' in globals() else 'N/A'))
            f.write("Output directory: {}\n".format(output_dir))
            f.write("Model path: {}\n".format(model_path))
            f.write("Number of unique knowledge point names: {}\n".format(len(knowledge_point_names)))
            f.write("Dense embedding shape: {}\n".format(dense_embeddings.shape))
            f.write("Sparse embedding count: {}\n".format(len(sparse_embeddings)))
            f.write("Processing time: {:.2f} seconds\n".format(processing_time))
            f.write("Timestamp: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S')))
            f.write("\nSample knowledge point names:\n")
            for i, name in enumerate(knowledge_point_names[:20]):
                f.write("  {}: {}\n".format(i+1, name))
            if len(knowledge_point_names) > 20:
                f.write("  ... and {} more\n".format(len(knowledge_point_names) - 20))
                
        logger.info("Saved processing log to {}".format(log_file))
        
    except Exception as e:
        logger.error("Error saving results: {}".format(e))
        raise


def process_embeddings_single_gpu(
    gpu_id,
    knowledge_point_names,
    output_dir,
    model_path,
    batch_size=256
):
    """
    Process embeddings on a single GPU
    
    Args:
        gpu_id: GPU device ID
        knowledge_point_names: List of knowledge point names to encode
        output_dir: Output directory
        model_path: Path to embedding model
        batch_size: Batch size for encoding
    """
    logger.info("Starting embedding extraction on GPU {}".format(gpu_id))
    logger.info("Processing {} unique knowledge point names".format(len(knowledge_point_names)))
    start_time = time.time()
    
    try:
        # Initialize model
        model = BGEM3FlagModel(
            model_path,
            use_fp16=True,
            pooling_method='cls',
            devices=["cuda:{}".format(gpu_id)]
        )
        logger.info("BGE-M3 model loaded on GPU {}".format(gpu_id))
        
        # Extract embeddings
        dense_embeddings, sparse_embeddings = bert_encode(
            model, knowledge_point_names, batch_size=batch_size
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info("Embedding extraction completed in {:.2f} seconds".format(processing_time))
        
        # Save results
        save_embeddings_and_metadata(
            dense_embeddings, sparse_embeddings, knowledge_point_names,
            output_dir, model_path, processing_time
        )
        
    except Exception as e:
        logger.error("Error in embedding extraction: {}".format(e))
        raise


def process_embeddings_multi_gpu(
    knowledge_point_names,
    output_dir,
    model_path,
    num_processes=1,
    gpu_num=1,
    batch_size=256
):
    """
    Process embeddings using multiple GPUs
    
    Args:
        knowledge_point_names: List of knowledge point names to encode
        output_dir: Output directory
        model_path: Path to embedding model
        num_processes: Number of processes per worker
        gpu_num: Number of GPUs available
        batch_size: Batch size for encoding
    """
    node_rank, nnodes = get_node_rank_and_node_nums()
    
    # Calculate data splits
    total_processes = num_processes * nnodes
    names_per_process = math.ceil(len(knowledge_point_names) / total_processes)
    
    logger.info("Multi-GPU processing: {} processes, {} GPUs".format(num_processes, gpu_num))
    logger.info("Names per process: {}".format(names_per_process))
    
    processes = []
    for i in range(num_processes):
        # Calculate data range for this process
        start_index = (node_rank * num_processes + i) * names_per_process
        end_index = min(
            (node_rank * num_processes + i + 1) * names_per_process,
            len(knowledge_point_names)
        )
        
        if start_index >= len(knowledge_point_names):
            break
            
        process_names = knowledge_point_names[start_index:end_index]
        gpu_id = i % gpu_num
        process_output_dir = "{}/process_{}_{}".format(output_dir, node_rank, i)
        
        logger.info("Process {}: GPU {}, names {}-{}".format(i, gpu_id, start_index, end_index))
        
        p = multiprocessing.Process(
            target=process_embeddings_single_gpu,
            args=(gpu_id, process_names, process_output_dir, model_path, batch_size)
        )
        processes.append(p)
    
    # Start all processes
    for p in processes:
        p.start()
    
    # Wait for completion
    for p in processes:
        p.join()
    
    logger.info("Multi-GPU processing completed")


def main():
    """Main function for embedding extraction"""
    # Get project root directory
    project_root = get_project_root()
    
    # Define default paths relative to project root
    default_input_file = project_root / "data" / "processed" / "knowledge_consolidation" / "consolidated_knowledge_points.txt"
    default_output_dir = project_root / "data" / "processed" / "embeddings"
    
    parser = argparse.ArgumentParser(description='Extract embeddings for knowledge point names')
    parser.add_argument('--input_file', type=str,
                       default=str(default_input_file),
                       help='Input consolidated knowledge points file')
    parser.add_argument('--output_dir', type=str,
                       default=str(default_output_dir),
                       help='Output directory for embeddings')
    parser.add_argument('--model_path', type=str,
                   default='./models/bge-m3',
                   help='Path to BGE-M3 model')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for encoding')
    parser.add_argument('--num_processes', type=int, default=1,
                       help='Number of processes per worker')
    parser.add_argument('--gpu_num', type=int, default=1,
                       help='Number of GPUs available')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Enable multi-GPU processing')
    
    global args
    args = parser.parse_args()
    
    # Log the paths being used
    logger.info("Project root: {}".format(project_root))
    logger.info("Input file: {}".format(args.input_file))
    logger.info("Output directory: {}".format(args.output_dir))
    
    # Check input file
    if not os.path.exists(args.input_file):
        logger.error("Input file not found: {}".format(args.input_file))
        return
    
    # Check model path
    if not os.path.exists(args.model_path):
        logger.error("Model path not found: {}".format(args.model_path))
        return
    
    # Extract knowledge point names
    logger.info("Extracting unique knowledge point names...")
    knowledge_point_names = extract_knowledge_point_names(args.input_file)
    
    if not knowledge_point_names:
        logger.error("No knowledge point names extracted")
        return
    
    # Process embeddings
    if args.multi_gpu and args.num_processes > 1:
        process_embeddings_multi_gpu(
            knowledge_point_names, args.output_dir, args.model_path,
            args.num_processes, args.gpu_num, args.batch_size
        )
    else:
        process_embeddings_single_gpu(
            0, knowledge_point_names, args.output_dir, args.model_path, args.batch_size
        )
    
    logger.info("Knowledge point names embedding extraction completed successfully!")
    logger.info("Next step: Use the generated embeddings for clustering analysis")


if __name__ == '__main__':
    main()
