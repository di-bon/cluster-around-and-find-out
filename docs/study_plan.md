# Problem statement

The current research aims at describing the viability and possible architecture of a converstaional clustering system.

The idea of this project is to shift the clustering processing from 'classical', i.e. by manually choosing a clustering algorithm (e.g. K-means, hierarchical clustering, DBSCAN)
and later finetuning its hyperparameters to achieve the expected results, to a more human-friendly approach, where the user describes in natural language what's the expected outcome
to an LLM and the system automatically clusters the data.
The system proposes a solution after each interaction with the user, and 2 results can occur:

1. the user accepts the result: the clustering system achieved a partition of the dataset that suits the user's needs. No more interaction is required.
1. the user requires some changes: the proposed clustering is not what the user expects, and additional changes are required (e.g. "split cluster 1", "merge clusters 2 and 5").
The clustering system, if needed, can ask some more clarifying questions to the user and then it proposes a new clustsering solution.

# Questions

Here are listed the main questions that the project **tries** to answer:

1. Is this approach viable? Is it better (easier to use with comparable results) than classical ones (e..g manually using clustering algorithms)?
1. What are some meaningful questions that the LLM should ask the user to gather as much information as possible to then cluster the data?
1. What should be the role of the LLM?
1. How should the clustering system be composed?
1. What should the LLM do and how should it cluster the data (i.e. choosing the optimal clustering algorithm and embedding representation)?
1. Will the system be able to process large datasets (e.g. hundreds of thousands of datapoints or more)?

# Simplifying assumptions

[ WIP ]
