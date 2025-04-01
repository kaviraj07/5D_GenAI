
# Embeddings:

- Embeddings are numerical representations of real-world data such as text, speech, image, or videos.
- The name embeddings refers to a similar concept in mathematics where one space can be mapped, or embedded, into another space.
- **Embeddings are expressed as low-dimensional vectors where the geometric distance between two vectors in the vector space is a projection of the relationship and semantic similarity between the two real-world objects that the vectors represent.**
- They provide compact representations of data of different types, while simultaneously also allowing you to compare two different data objects and tell how similar or different they are on a numerical scale.
- Help in efficient largescale data processing and storage by acting as means of lossy compression of the original data while retaining its important semantic properties.

## Embedding in world of multimodality

- Joint embeddings are when multiple types of objects are being mapped into the same embeddings space, for example retrieving videos based on text queries.
- These embedding representations are designed to capture as much of the original object’s characteristics as possible.

![](Screenshots/joint_embedding.png)

## Evaluating Embedding Quality

Two important metrics for evaluating quality:

1. Precision - all documents retrieved should be relevant
2. Recall - all of the relevant documents should be retrieved
3. Precision at K (for only top k results)
4. Recall at K (for only top k results)
5. Normalized Discounted Cumulative Gain (nDCG) - use when order of results matter - the quality of the ranking produced by the embedding model compared to the desired ranking. (A top result on google page 2 is not actually relevant)

$$DCG= \sum_{i=1}^{P} \frac{rel_i}{log_2(i+1)} , \quad \text{where }p \text{ is position and } rel_i \text{ is the relevance score}$$

## Search Example

### Retrieval Augmented Generation (RAG)

Two main processes:

1. Index creation where documents are divided into chunks which are used to generate embeddings and stored in a vector database for low latency searches.
2. When the user asks a question to the system that is embedded using the query portion of the model and which will map to relevant documents when using a similarity search in the vector database.


![](Screenshots/RAG.png)
## Types of Embeddings

### Text Embedding

- Used extensively for NLP
- Embed meaning of natural language
- Two broad categories:
	- Token/word
	- Document


#### Lifecycle:

![](Screenshots/text-life-cycle.png)

- Text: Input string
- Tokenization: Input string is split into smaller meaningful pieces called tokens.
- Indexing: Each of these tokens is then assigned a unique integer value usually in the range: \[0, cardinality of the total number of tokens in the corpus].
- Embedding: The output this indexing is called embedding.

### Word Embedding

Common word embeddings:
1. GloVe
2. SWIVEL
3. Word2Vec

#### Word2Vec

Word2Vec is a family of model architectures that operates on the principle of “the semantic meaning of a word is defined by its neighbors”, or words that frequently appear close to each other in the training corpus.

Types of Word2Vec Architectures:

1. Continuous bag of words (CBOW): Tries to predict the middle word, using the embeddings of the surrounding words as input. (faster to train, suitable for bigger datasets)
2. Skip-gram: The setup is inverse of that of CBOW, with the middle word being used to predict the surrounding words within a certain range. (slower to train, suitable for smaller datasets)

![](Screenshots/cbow_skip_gram.png)

- FastText is an extention of word2vec by going into sub-word level.

- Cons of word2vec: does not capture global statistics (words in whole corpus)

#### GloVe

GloVe is a word embedding technique that leverages both global and local statistics of words.

Process:

- Create a co-occurrence matrix for representing the relationships between words.
- Factorization technique to learn those representations from the matrix.
- Result of the factorization captures both global and local information.


#### Swivel

- SWIVEL is another approach which leverages the co-occurrence matrix to learn word - embeddings. 
- SWIVEL stands for Skip-Window Vectors with Negative Sampling.

Process:
- uses local windows to learn word vectors from the co-occurrence matrix
- SWIVEL also considers unobserved co-occurrences and handles it using a special piecewise loss, boosting its performance with rare words.
- Less accurate than GloVe but faster to train.


### Document Embedding

Document Embeddings is about representing large chunks of text or documents.

Two categorized stages:
- Bag-of-words (BoW)
- Deeper trained LLMs


#### Bag of Words

- Latent semantic analysis (LSA) uses a co-occurrence matrix of words in documents.
- Latent dirichlet allocation (LDA) uses a bayesian network to model the document embeddings.
- TF-IDF (term frequency-inverse document frequency) based models, which are statistical models that use the word frequency to represent the document embedding.

**Drawback:** 1. Word ordering and 2. Semantic meanings are ignored.

**Solution:** Doc2Vec model adds an additional ‘paragraph’ embedding or, in other words, document embedding in the model of Word2Vec. The paragraph embedding is concatenated or averaged with other word embeddings to predict a random word in the paragraph.


#### Deeper pretrained large language models

- BERT - bidirectional encoder representations from transformers was proposed with groundbreaking results on 11 NLP tasks in 2018
- PaLM
- Gemini
- GPT
- Llama
- GTR
- Sentence-T5
- Matryoshka Embeddings

>Although the deep neural network models require a lot more data and compute time to train, they have much better performance compared to models using bag-of-words paradigms.


![](Screenshots/taxonomy_of_emb_mod.png)


### Image and Multimodal Embeddings

**Unimodal Embedding** - Training a CNN or Vision Transformer model on a large scale image classification task (for example, Imagenet), and then using the penultimate layer as the image embedding.

**Multimodal Embedding** - take the individual unimodal text and image embeddings and create the joint embedding of their semantic relationships learnt via another training process.


### Structured data embeddings


- Structured data refers to data has a defined schema, like an table in a database where individual fields have known types and definitions.

- We have to create the embedding model for the structured data since it would be specific to a particular application.

#### General Structured Data

- Done using ML models in the dimensionality reduction category, such as the PCA model
- Use case 1: anomaly detection
- Use case 2: Downstream ML task such as classification

#### User/item Structured Data

- The input is no longer a general structured data table as above.
- This category is for recommendation purposes, as it maps two sets of data (user dataset, item/product/etc dataset) into the same embedding space.

### Graph Embeddings

- Graph embeddings are another embedding technique that lets you represent not only information about a specific object but also its neighbors (namely, their graph representation).

- Popular algorithms for graph embedding include DeepWalk, Node2vec, LINE, and GraphSAGE.

### Training Embeddings

- Embedding models use dual encoder (two tower) architecture.
- For example, for the text embedding model used in question-answering, one tower is used to encode the queries and the other tower is used to encode the documents.
- The loss used in embedding models training is usually a variation of contrastive loss, which takes a tuple of <inputs, positive targets, \[optional] negative targets> as the inputs. Training with contrastive loss brings positive examples closer and negative examples far apart.
- Training includes two stages: pretraining (unsupervised learning) and fine tuning (supervised learning).
- Pre-training can be skipped by leveraging already pre-trained foundational model such as BERT, T5, GPT, Gemini and CoCa.
- Fine-tuning is done is one or several phases: various methods, including human labelling, synthetic dataset generation, model distillation, and hard negative mining.

# Vector Search

- It uses the vector or embedded semantic representation of documents.
- As vector search works on any sort of embedding it also allows search on images, videos, and other data types in addition to text.
- Vector search lets you to go beyond searching for exact query literals and allows you to search for the meaning across various data modalities.