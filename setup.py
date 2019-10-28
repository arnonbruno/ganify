import setuptools

# The directory containing this file
with open("README.md", "r") as fh:
    long_description = fh.read()


# This call to setup() does all the work
setuptools.setup(
    name="ganify",
    version="1.0.10",
    description="An Easy way to use GANs for data augmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnonbruno/ganify",
    author="Arnon Bruno",
    author_email="asantos.quantum@gmail.com",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['tensorflow>=2.0.0', 'pandas>=0.25',
                      'numpy>=1.16', 'scikit-learn>=0.21', 'matplotlib>=3.1',
                      'tqdm>=4.15'],
)
