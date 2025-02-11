from setuptools import setup, find_packages

setup(
    name="molecular_representations",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib>=3.7.1",
    ],
)
