"""
CogAtom基础使用示例
"""
from cogatom import CogAtomPipeline

def main():
    # 初始化pipeline
    pipeline = CogAtomPipeline("configs/config.yaml")
    
    # 使用示例数据运行
    results = pipeline.run_full_pipeline(
        seed_data_path="data/samples/sample_seed.jsonl",
        num_problems=50
    )
    
    print("生成完成！")
    print(f"提取知识点数量: {results['knowledge_points']}")
    print(f"知识图节点数: {results['graph_nodes']}")
    print(f"知识图边数: {results['graph_edges']}")
    print(f"生成问题数量: {results['generated_problems']}")
    print(f"高质量问题数量: {results['high_quality_problems']}")
    print(f"输出文件: {results['output_file']}")

if __name__ == "__main__":
    main()
