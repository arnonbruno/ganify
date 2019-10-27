import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="ganify",
    version="1.0.1",
    description="An Easy way to use GANs for data augmentation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/arnonbruno/ganify",
    author="Arnon Bruno",
    author_email="asantos.quantum@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['tensorflow>=2.0.0', 'pandas>=0.25',
                      'numpy>=1.16', 'scikit-learn>=0.21', 'matplotlib>=3.1'],
)
