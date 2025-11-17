#!/usr/bin/env python
"""
Setup script for GIFTpy package.

For modern Python packaging, see pyproject.toml.
This setup.py is maintained for backward compatibility.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version
version = "0.1.0"

setup(
    name="giftpy",
    version=version,
    description="Geometric Information Field Theory - Python package for unified physics predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GIFT Framework Collaboration",
    author_email="gift@gift-framework.org",
    url="https://github.com/gift-framework/GIFT",
    project_urls={
        "Documentation": "https://giftpy.readthedocs.io",
        "Source": "https://github.com/gift-framework/GIFT",
        "Issues": "https://github.com/gift-framework/GIFT/issues",
    },
    packages=find_packages(include=["giftpy", "giftpy.*"]),
    package_data={
        "giftpy": ["data/**/*"],
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.4.0",
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "ipython>=8.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="physics theoretical-physics particle-physics standard-model cosmology",
    license="MIT",
    zip_safe=False,
)
