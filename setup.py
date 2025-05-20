"""
Setup script for Nexus 1.1
"""

from setuptools import setup, find_packages

setup(
    name="nexus-1-1",
    version="0.1.0",
    description="Nexus 1.1 - Advanced Autonomous AI Model",
    author="OpenHands",
    author_email="openhands@all-hands.dev",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "gradio>=3.32.0",
        "requests>=2.28.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)