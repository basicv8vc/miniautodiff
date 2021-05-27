import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="miniad",
    version="0.1.1",
    author="basicv8vc",
    author_email="basicv8vc@gmail.com",
    description="Another mini autograd engine with a PyTorch-like API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/basicv8vc/miniautodiff",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
