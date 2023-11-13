# bertopic_modeling
Topic Modeling as unsupervised learning problem (hard clustering of short-text using BERTopic).

## The Idea

Fine-tuning the hyperparameters of UMAP and HDBSCAN algortihms to perform hard clustering of short text in italian language.
The BERTopic pipeline allows to easily build a topic modeling pipeline creating dense clusters and interpretable topic descriptions. By default, the main steps for topic modeling with BERTopic are sentence-transformers, UMAP, HDBSCAN, and c-TF-IDF run in sequence. 
The steps chosen in the context of this work were:
  1. Google’s Universal Sentence Encoder (USE), first published by Cer et al in 2018, which is a popular sentence embedding model that performs well on sentence semantic similarity tasks within one language or multiple languages. Its multilingual version was trained on italian corpus.
  2. UMAP for reducing deìimensionality of embeddings.
  3. HDBSCAN as density-based algorithm and it doed not require specifying the number of clusters upfront is indifferent to cluster shape.
  4. CountVectorizer
  5. KeyBERTInspired representation model which performs fine-tuning based on the semantic relationship between keywords/keyphrases and the set of documents in each topic.

UMAP has several hyperparameters that control how it performs dimensionality reduction, but two of the most important are **n_neighbors** and **n_components**. The n_neighbors parameter controls how UMAP balances local versus global structure in the data. This parameter controls the size of the neighborhood UMAP looks to learn the manifold structure, and so lower values of n_neighbors will focus more on the very local structure. The n_components parameter controls the dimensionality of the final embedded data after performing dimensionality reduction on the input data.

HDBSCAN also has several important hyperparameters, but the most important one to consider is **min_cluster_size**. Intuitively, this controls the smallest grouping you want to consider as a cluster. 


## Setup
Install this module using

`pip install git+https://github.com/ceciliacasarella/bertopic_modeling.git`

**Start the mlflow tracking server**[mlflow-doc]("https://mlflow.org/docs/latest/tracking.html")

`mlflow server`

The MLflow Tracking component is an API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results. 
MLflow Tracking is organized around the concept of **runs**, which are executions of some piece of data science code.
By default, the MLflow Python API logs runs locally to files in an mlruns directory wherever you ran your program.

`python bertopic_optimizer.py --experiment-name BERT_Topic_exp --run_name bert-topic-first-exp-1  --tracking_client http://127.0.0.1:5000 --label_lower 5 --label_upper 15 --penalty 0.3 --max_evals 1 --data_path C:\\path-to-file`



