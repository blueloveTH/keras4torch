import setuptools

with open("README.md", "rt", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras4torch",
    version="0.3.1",
    author="Keras4Torch Development Team",
    author_email="blueloveTH@foxmail.com",
    description="An Easy-to-Use Wrapper for Training PyTorch Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/blueloveTH/keras4torch",
    packages=setuptools.find_packages(),
    install_requires=['torch-summary==1.4.3', 'pandas'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)