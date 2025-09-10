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
