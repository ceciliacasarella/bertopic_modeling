from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Short text topic modeling'
LONG_DESCRIPTION = 'HDBSCAN and UMAP hyperparameters tuning for short text topic modeling by means of BERTopic'

# Setting up
setup(
    name="bertopic.modeling",
    version=VERSION,
    author="Cecilia Casarella",
    author_email="<casarellacecilia@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    readme="README.md",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'nltk','tensorflow','tensorflow_text','tensorflow_hub','umap','umap-learn','hdbscan','hyperopt','mlflow','bertopic'],
    keywords=['python', 'BERTopic', 'short text', 'sentence encoding', 'italian survey', 'hyperparameters tuning', 'hdbscan','umap'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)