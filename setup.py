from setuptools import setup, find_packages

setup(
    name="array_processing",
    version="0.1.0",
    description="A library for DOA estimation methods and related techniques",
    author="Marco Marinho",
    packages=find_packages(include=["array_processing", "array_processing.*"]),
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "matplotlib>=3.7.1"
    ],
    extra_require=["jupyter"]
)
