#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

def get_install_requires() -> str:
    return [
        "tqdm",
        "numpy",
        "open3d",
        "gpytorch",
        "matplotlib",
        "ray", 
        "pandas",
        "seaborn",
        "dcargs"
    ]

setup(
    name="roller_slam",
    version='0.0.1',
    description="A Library for Inhand SLAM",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="Chaoyi Pan",
    author_email="jc-bao@outlook.com",
    license="MIT",
    python_requires=">=3.6",
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    install_requires=get_install_requires(),
)
