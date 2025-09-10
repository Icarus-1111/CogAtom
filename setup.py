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
