import re
from pathlib import Path

import setuptools

root = Path(__file__).parent
version = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
    (root / "ganify" / "_version.py").read_text(encoding="utf-8"),
).group(1)
long_description = (root / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="ganify",
    version=version,
    description="An easy way to use GANs for tabular data augmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnonbruno/ganify",
    author="Arnon Bruno",
    author_email="asantos.quantum@gmail.com",
    packages=setuptools.find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    package_data={
        "benchmarks": [
            "configs/gates/*.yaml",
            "configs/models/*.yaml",
            "configs/releases/*.yaml",
            "configs/suites/*.yaml",
            "manifests/datasets/*.yaml",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "tensorflow>=2.2",
        "pandas>=0.25",
        "numpy>=1.16",
        "scikit-learn>=0.21",
        "matplotlib>=3.1",
        "tqdm>=4.15",
    ],
)
