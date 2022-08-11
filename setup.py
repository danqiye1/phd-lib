import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-patterns", # Replace with your own username
    version="0.0.1",
    author="Ye Danqi",
    author_email="yedanqi@comp.nus.edu.sg",
    description="Library of patterns for PyTorch deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
)