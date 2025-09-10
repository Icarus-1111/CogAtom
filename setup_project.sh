#!/bin/bash

echo "创建CogAtom项目结构..."

# 创建主包目录
mkdir -p cogatom

# 创建数据目录结构
mkdir -p data/raw/seed_problems
mkdir -p data/raw/knowledge_base
mkdir -p data/processed/knowledge_points
mkdir -p data/processed/graphs
mkdir -p data/processed/embeddings
mkdir -p data/generated/problems
mkdir -p data/generated/solutions
mkdir -p data/generated/metadata
mkdir -p data/evaluation/quality_assessment
mkdir -p data/evaluation/difficulty_analysis
mkdir -p data/evaluation/diversity_analysis
mkdir -p data/samples

# 创建其他目录
mkdir -p configs
mkdir -p examples
mkdir -p tests
mkdir -p scripts

# 创建主包文件
cat > cogatom/__init__.py << 'INIT'
"""
CogAtom: Cognitive Atom for Mathematical Problem Generation
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .pipeline import CogAtomPipeline

__all__ = ["CogAtomPipeline"]
INIT

# 创建基础Python文件
touch cogatom/knowledge_extraction.py
touch cogatom/graph_construction.py
touch cogatom/problem_generation.py
touch cogatom/quality_evaluation.py
touch cogatom/pipeline.py
touch cogatom/utils.py

# 创建配置文件
cat > configs/config.yaml << 'CONFIG'
# CogAtom配置文件

# 知识提取配置
knowledge:
  model: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"  # 从环境变量读取
  batch_size: 10

# 图构建配置  
graph:
  clustering_method: "leiden"
  min_edge_weight: 0.1
  co_occurrence_window: 5

# 问题生成配置
generation:
  model: "qwen2.5-72b"
  api_key: "${QWEN_API_KEY}"
  max_problems_per_batch: 100
  sampling_strategy: "weighted_random_walk"
  temperature: 0.7

# 质量评估配置
evaluation:
  model: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"
  quality_threshold: 0.7
  batch_size: 20
CONFIG

# 创建requirements.txt
cat > requirements.txt << 'REQ'
openai>=1.0.0
pyyaml>=6.0
numpy>=1.21.0
pandas>=1.3.0
networkx>=2.6.0
scikit-learn>=1.0.0
tqdm>=4.62.0
REQ

# 创建setup.py
cat > setup.py << 'SETUP'
from setuptools import setup, find_packages

setup(
    name="cogatom",
    version="0.1.0",
    description="Cognitive Atom: Knowledge-driven mathematical problem generation",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "pyyaml>=6.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
)
SETUP

# 创建基础使用示例
cat > examples/basic_usage.py << 'EXAMPLE'
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
EXAMPLE

# 创建示例数据
cat > data/samples/sample_seed.jsonl << 'SAMPLE'
{"id": "sample_1", "problem": "求解方程 x^2 + 3x - 4 = 0", "solution": "使用求根公式，x = (-3 ± √(9+16))/2 = (-3 ± 5)/2，所以 x = 1 或 x = -4", "difficulty": "medium", "topics": ["代数", "二次方程"]}
{"id": "sample_2", "problem": "计算三角形面积，已知三边长为3, 4, 5", "solution": "这是直角三角形，面积 = (1/2) × 3 × 4 = 6", "difficulty": "easy", "topics": ["几何", "三角形"]}
{"id": "sample_3", "problem": "求函数 f(x) = x^3 - 3x + 1 在 x = 2 处的导数", "solution": "f'(x) = 3x^2 - 3，所以 f'(2) = 3×4 - 3 = 9", "difficulty": "medium", "topics": ["微积分", "导数"]}
SAMPLE

# 创建基础测试
cat > tests/test_basic.py << 'TEST'
"""
基础测试
"""
import unittest
from pathlib import Path

class TestCogAtom(unittest.TestCase):
    
    def test_import(self):
        """测试包导入"""
        try:
            from cogatom import CogAtomPipeline
            self.assertTrue(True)
        except ImportError:
            self.fail("无法导入CogAtomPipeline")
    
    def test_data_structure(self):
        """测试数据目录结构"""
        data_dir = Path("data")
        self.assertTrue(data_dir.exists())
        self.assertTrue((data_dir / "samples").exists())
        self.assertTrue((data_dir / "samples" / "sample_seed.jsonl").exists())

if __name__ == "__main__":
    unittest.main()
TEST

# 创建.gitignore
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 数据文件（除了samples）
data/raw/
data/processed/
data/generated/
data/evaluation/
!data/samples/

# 配置文件中的敏感信息
configs/local_config.yaml

# IDE
.vscode/
.idea/
*.swp
*.swo

# 日志
*.log
logs/

# 临时文件
*.tmp
*.temp

# 环境变量文件
.env
GITIGNORE

# 创建README.md
cat > README.md << 'README'
# CogAtom

**Cognitive Atom**: Knowledge-driven mathematical problem generation pipeline.

## 🚀 Quick Start

```bash
# 安装
pip install -e .

# 基础使用
python examples/basic_usage.py
📁 Project Structure
cogatom/
├── cogatom/           # 主包
├── data/              # 数据目录
│   ├── raw/          # 原始数据
│   ├── processed/    # 处理后数据
│   ├── generated/    # 生成数据
│   └── samples/      # 示例数据
├── configs/          # 配置文件
├── examples/         # 使用示例
└── tests/           # 测试文件
🔧 Configuration
在 configs/config.yaml 中设置API密钥和模型参数。

📊 Pipeline Steps
Knowledge Extraction: 从种子问题中提取知识点
Graph Construction: 构建知识点关系图
Problem Generation: 基于知识图生成新问题
Quality Evaluation: 评估生成问题的质量
🤝 Contributing
欢迎贡献代码和建议！

📄 Citation
如果使用本项目，请引用我们的论文：[论文信息] README

echo "项目结构创建完成！"
