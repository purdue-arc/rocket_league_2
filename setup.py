#!/usr/bin/env python3
"""
Setup script for Rocket League 2 project
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

# Read README for long description
def read_readme():
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            return f.read()
    return "Rocket League 2 - Teaching cars soccer"

setup(
    name="rocket_league_2",
    version="1.0.0",
    description="Autonomous Rocket League robot training system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Purdue ARC",
    author_email="adarshveerapaneni@gmail.com",
    url="https://github.com/purdue-arc/rocket_league_2",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "training": [
            "stable-baselines3[extra]>=2.0.0",
            "tensorboard>=2.8.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "perception": [
            "ultralytics>=8.0.0",
            "opencv-python>=4.5.0",
            "robotpy-apriltag>=0.4.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "rocket-league-train=rktl_autonomy.train_sb3:main",
            "rocket-league-sim=rktl_simulator.simulator:main",
        ],
    },
)
