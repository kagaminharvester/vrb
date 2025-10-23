"""
Setup script for VR Body Segmentation Application
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="vr-body-segmentation",
    version="1.0.0",
    author="VR Body Segmentation Team",
    description="Real-time body segmentation for VR videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vr-body-segmentation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "tensorrt": [
            "tensorrt>=8.6.0",
        ],
        "onnx": [
            "onnx>=1.14.0",
            "onnxruntime-gpu>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vr-segment=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
