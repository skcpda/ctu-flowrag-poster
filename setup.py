#!/usr/bin/env python3
"""
Setup script for CTU-FlowRAG package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("ctu_flowrag/README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ctu-flowrag",
    version="1.0.0",
    author="CTU-FlowRAG Team",
    author_email="team@ctu-flowrag.com",
    description="Role-Conditioned Retrieval with Graph Attention Networks and Capacitated Soft Role Assignment",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ctu-flowrag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "matplotlib>=3.5.0",
            "networkx>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ctu-flowrag-prepare=ctu_flowrag.scripts.prepare_tensors:main",
            "ctu-flowrag-train=ctu_flowrag.train.train_rcr_gat:main",
            "ctu-flowrag-eval=ctu_flowrag.eval.eval_paths:main",
            "ctu-flowrag-viz=ctu_flowrag.viz.plot_ctu_graph:main",
            "ctu-flowrag-attention=ctu_flowrag.retrieval.attention_inspector:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

