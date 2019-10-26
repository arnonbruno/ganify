import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="GANify",
    version="1.0.0",
    description="An Easy way to use GANs for data augmentation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/arnonbruno/ganify",
    author="Arnon Bruno",
    author_email="asantos.quantum@gmail.com",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["Tensorflow 2.0", "Pandas",
                      'Numpy', 'scikit-learn', 'matplotlib'],
)
