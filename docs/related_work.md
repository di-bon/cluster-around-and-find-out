# Related work

This document contains a brief summary of some papers about clustering and user interaction in natural language.

## ClusterLLM

ClusterLLM leverages LLMs (in this case `gpt-3.5-turbo`) to:

- gather insights on clustering perspectives by asking hard triplet questions (e.g. "does A better correspond to B than C?")
- derive the cluster granularity by asking pairwise questions (e.g. "do A and B belong to the same category")

A limitation of the paper is represented by the fact that the authors use API-based LLMs, so the work focuses on reducing the average cost per dataset.
More recent, locally-hosted models (such as qwen3.6) were not tested in this paper.

### LLM usage

The paper identifies 2 interaction stages.

The first stage aims at finding 3 different datapoints to identify user's preferences (does A better correspond to B than C?).
Entropy is used as the metric to choose A, B and C (samples with highest entropy).

The second stage aims at finding the optimal number of clusters.
In this stage, hierarchical clustering methods are used.
After having clustered the entire dataset, the LLM is used to choose the granularity of the clusters, by relying on few annotated data pairs as demonstration.

### Experiments

Two main evaluation methods are used:

- clustering accuracy (ACC) by first using the Hungarian algorithm
- normalized mutual information (NMI)

Paper: https://aclanthology.org/2023.emnlp-main.858.pdf.

## Large Language Models Enable Few-Shot Clustering

It applies LLM interaction before, during and after clustering, to improve the results.
The interaction with the LLM aims at finding pairwise constraints.

The 3 possible interactions:

1. before clustering: the LLM generates a keyphrase for each sample and adds it to the base representation
1. during clustering: clustering constraints are added
1. after clustering: low-confidence cluster assignments are corrected by the user by providing some pairwise constraints

The first 2 methods are the most effective ones.

It's worth noticing that the methods 1 and 3 can be applied to any text clustering algorithm using any set of text features!
This shows how versatible and flexible these methods are.

The model used in this paper is `gpt-3.5-turbo-0301`.

### Keyphrase generation

Each sample is fed to an LLM before the clustering algorithm is applied.
The output of this preprocessing step is a JSON file with a set of keyphrases that can describe the datapoint.
The keyphrase set is then encoded into a vector along with the original document's text representation and then all of this is given as input to an encoder to generate the embedding for the augmented datapoint.

### Clustering constraints

When the clustering algorithm is applied, it leverages a set of pairwise constraints defined by an expert (or the end user), so that pairs of points must or cannot be lined together.
During this step, the LLM is used as pseudo-oracle to amplify the expert guidance.
The pairs of points that need to be classified are chosen based on metrics such as the Explore-Consolidate algorithm or the euclidean distance between embeddings, depending on the number of embeddings present.
The LLM is also used to generate pairwise constraint by itself, given some examples created by the expert.

### Cluster correction

The points with the lowest confidence (i.e. the *k* points with the least margin between the nearset and second-nearest clusters).
The LLM then checks whether these points are correctly classified or if a different assignment is required.
The new assignemt is chosen among the 4 nearest clusters.

### Evaluation

To evaluate the results of entity canonicalization, the metrics used are:

- Macro precision and recall
- Micro precision and recall
- Pairwise precision and recall

Then, the armonic mean of these 3 metrics is computed.

To evaluate the results of text clustering, the cluster assignemnts are compared to the ground truth using normalized mutual information and accuracy, also by applying the Hungarian algorithm.
Text clustering results do not match the state of the art approaches (SCCL and ClusterLLM), but they're not too far either.

Paper: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00648/120476

## Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues

This paper tries to improve the results presented by ClusterLLM, by focusing more on the semantic meaning of the text that is being clustered.
The novelty of the proposed approach is the LLM-in-the-loop, which means that a fine-tuned LLM is used during the clustering process as a judge.

The 4 proposed steps are:

1. coherence evaluation: 
it is a more effective metric and optimization objective, used to measure the semantic consistency within a cluster.
It is formulated as a binary classification problem, where an LLM evaluator assigns a label (either 'good' or 'bad') to each cluster, based on the information they contain.
Specific clusters will receive the 'good' label, while clusters that result inconsistent or with ambiguous intentions will receive the 'bad' label.
Also, a name is assigned to each cluster, to better state what each cluster represent.

1. LLM-in-the-loop iterative intent clustering with coherence evaluation:
by leveraging the 'good' and 'bad' labels, a local search algorithm can be used to discover the optimal number of clusters at each iteration.
This step relies on K-means.
After each iteration, the clusters identified as 'good' are removed from the dataset and the same algorithm is then applied to the remaining datapoints.
This loop is repeated until only few datapoints remain unassigned to good clusters.

1. post-correction with LLM-generated intent labels:
the previous step might generate, at different iterations, multiple clusters that capture similar intents.
Other papers rely on using an LLM to validate the obtained clusters, but this approach is computationally expensive and pronce to inconsistency.
The proposed method is a context-aware approach, leveraging the LLM's naming utility to merge clusters based on their generated intent labels.

The intent labels are first encoded and represented by an embedding, and later the distance between the embeddings is used find intent similarity.
Note that the measurement of semantic relationship is preformed on a sphere surface rather than straight-line distance, to better deal with the high-dimensional space that contains the embeddings.

After this step, the clusters are represented with an affinity graph, and the edges are created based on a thresholded geodisc distance.

1. context-aware role separation with LLM-generated intent labels:
in the context of customer service calls, we can identify 2 subjects, the customer and the service agent.
By using the intent label generated by the LLM, we can refine the clusters and assign datapoints to one of the two possible subjectes that take part in the conversation.
Doing this, we can re-cluster the data of each cluster to have cleaner results.

Paper: https://aclanthology.org/2025.emnlp-main.300.pdf
