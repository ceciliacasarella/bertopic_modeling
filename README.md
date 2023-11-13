# bertopic_modeling
Topic Modeling as unsupervised learning problem (hard clustering of short-text using BERTopic).

<h2>The Idea</h2>

Fine-tuning the hyperparameters of UMAP and HDBSCAN algortihms to perform hard clustering of short text in italian language.
The BERTopic pipeline allows to easily build a topic modeling pipeline creating dense clusters and interpretable topic descriptions. By default, the main steps for topic modeling with BERTopic are sentence-transformers, UMAP, HDBSCAN, and c-TF-IDF run in sequence. 
The steps chosen in the context of this work were:
  1. Google’s Universal Sentence Encoder (USE), first published by Cer et al in 2018, which is a popular sentence embedding model that performs well on sentence semantic similarity tasks within one language or multiple languages. Its multilingual version was trained on italian corpus.
  2. UMAP
  3. HDBSCAN
  4. 
