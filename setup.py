import setuptools
import keras4torch

with open("README.md", "rt", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras4torch",
    version=keras4torch.__version__,
    author="blueloveTH",
    author_email="blueloveTH@qq.com",
    description="A Ready-to-Use Wrapper for Training PyTorch Models✨",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/blueloveTH/keras4torch",
    packages=setuptools.find_packages(exclude=('minimum')),
    install_requires=['pandas'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)