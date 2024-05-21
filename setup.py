import setuptools
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "1.8.1.1"
DESCRIPTION = "To be added in the future"


setuptools.setup(
    name="ise_cdg_data",
    version=VERSION,
    author="Ashkan Khademian",
    author_email="ashkan.khd.q@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "torchtext>=0.15.0",
        "spacy>=3.6.0",
        "pygments>=2.0.0",
        "pandas>=1.5.0",
        "networkx>=3.0",
        "markdown>=3.5",
        "beautifulsoup4>=4.12",
        "tqdm>=4.0.0",
        "radon",
        # "gensim==3.8.3", # Not Installable on python 3.10 thus ruining Google Colab
        "sumy==0.11.0",
        "sentence-transformers",
        "rank_bm25",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
