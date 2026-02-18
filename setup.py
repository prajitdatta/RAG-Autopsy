from setuptools import setup, find_packages

setup(
    name="rag-autopsy",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=8.0.0",
    ],
)
