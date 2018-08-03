# SOM

## Input Description

The dataset consists on text documents from Brasiliean companies classified in 9 categories. The original documents were pre-processed in order to obtain a Bag-Of-Words style. In this format each document is represented as a vector where each dimension corresponds to a specific word and its value represents the frecuency of that word within the document. Articles and prepositions were filtered.
The data contains 900 entries uniformly distributed across the 9 categories. Each entry represent a document and is made of 1 attribute showing the category number (1 to 9) and 850 features corresponded to the frecuency of words. This is a highly sparse set (> 99\% are ceros).
The problem consists in classify each document by using 2 non-supervised techniques. The information about the category number of each document is only used to evaluate the final classification score.

### Ej1: Dimensionality Reduction (PCA)

We built a hebbian learning algorithm NN able to reduce the high dimensionality problem (850) in 3 dimensions. Sanger's and Oja's learning rules were used in order to obtain the principal component vectors.

### Ej2: Self-Organizing Map (SOM)

We built a 2-D grid which maps the features of the inputs in a clustered representation for visualization. The algorithm is based on a competitive learning, an example of a Kohonen application.
