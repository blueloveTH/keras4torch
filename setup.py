import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras4torch",
    version="0.1.2",
    author="keras4torch Development Team",
    author_email="blueloveTH@foxmail.com",
    description="A keras-like high level API for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/blueloveTH/keras4torch",
    packages=setuptools.find_packages('keras4torch'),
    install_requires=['torch-summary==1.4.3', 'pandas', 'numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)