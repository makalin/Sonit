#!/usr/bin/env python3
"""
Setup script for Sonit
Translating the Unspoken
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sonit",
    version="0.1.0",
    author="makalin",
    author_email="makalin@gmail.com",
    description="Translating the Unspoken - A vocal gesture translator",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/makalin/Sonit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sonit=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.kv", "*.png", "*.jpg", "*.json"],
    },
    keywords="audio, machine-learning, accessibility, vocal-gestures, translation",
    project_urls={
        "Bug Reports": "https://github.com/makalin/Sonit/issues",
        "Source": "https://github.com/makalin/Sonit",
        "Documentation": "https://github.com/makalin/Sonit#readme",
    },
) 