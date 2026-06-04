# Related work

v3 - 2026-06-04

This document contains a brief summary of some papers about clustering and user interaction in natural language.

## Changelog

### v3 (2026-06-04)
- Papers read and added:

    - Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models

### v2 (2026-05-28)

- Papers read and added:
    
    - NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models

### v1 (2026-05-21)

- Papers read and added:

    - Optimized Algorithms for Text Clustering with LLM-Generated Constraints
    - Eliciting Human Preferences with Language Models
    - TO-GATE: Clarifying Questions and Summarizing Responses with Trajectory Optimization for Eliciting Human Preference
    - Summaries as Centroids for Interpretable and Scalable Text Clustering
    - LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering
    - **Asking Clarifying Questions for Preference Elicitation With Large Language Models**
    - **Improving Text Embeddings with Large Language Models**

- Add takeaways to every paper
- Add "What we don't know yet about the field" section

### v0 (2026-05-14)

- Papers read and added:

    - ClusterLLM: Large Language Models as a Guide for Text Clustering
    - Large Language Models Enable Few-Shot Clustering
    - Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues

### What we don't know yet about the field

Lots of research focuses on clustering by assuming that a ground-truth/underlying 'correct' assignment exists.
The project assumes that no correct cluster assignment exists, as the solution is highly subjective.
Unfortunately, no work on this specific topic was found, meaning that the evaluation of the final results can only focus on the conversational aspect of the process;
the clustering assignment cannot be evaluated by neither internal nor external metrics, but other studies may be able to identify appropriate metrics (especially internal ones).

## Methodogical precedents

### LLM conversation

A conversation with an LLM proved to be an effective method to gather insights on users' preferences, as showed in recent works [1, 6, 7].
Moreover, specifically fine-tuned models can derive users' preferences [9].
Because of this, the user-facing LLM-model of this project should follow the same concept to derive the clustering preferences of the users.

### Embeddings generation

Versatile encoder models can take into account problem-specific characteristics to create context-coherent embeddings [10].
Thus, the encoder model should take as input the summary of the users' preferences.

### Clustering algorithms

Clustering algorithms can leverage LLMs to improve their results, as showed in recent papers [2, 3, 4, 7].
For this reason, methods such as keyprhase generation or summaries-of-centroids must be investigated to assess their benefits.

# Papers summaries

## 1. ClusterLLM: Large Language Models as a Guide for Text Clustering

Paper: https://aclanthology.org/2023.emnlp-main.858.pdf

**Takeways**: ask triplets or pairwise questions.

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

## 2. Large Language Models Enable Few-Shot Clustering

Paper: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00648/120476

**Takeaways**: keyphrase generation (before clustering), ML/CL constraints, geenrated also by the LLM (during clustering).

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

## 3. Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues

Paper: https://aclanthology.org/2025.emnlp-main.300.pdf

**Takeaways**: LLM evaluator to assign 'good' or 'bad' labels to clusters. 
Post-correction to merge clusters with similar intents.

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

## 4. Optimized Algorithms for Text Clustering with LLM-Generated Constraints

Paper: https://arxiv.org/pdf/2601.11118

**Takeaways**: CL/ML constraints are sets rather than pairwise relations.

This paper proposes a constraint-generation approach that reduces resource consumption by generating constraint sets rather than using traditional pairwise constraints.
The results are comparable to state-of-the-art algorithms, but with a reduction of LLM queries of over 20 times.

The work focuses on genearting cannot-link (CL) and must-link (ML) constraints using an LLM.

### Constraing generation with LLM

Candidate constraint sets for CL and ML constraints need to contain at least 2 points per query.
Candidate points are selected based on distance.

#### Must-Link constraint set

Candidate points are identified by first applying a coreset-based algorithm, then each subset is mapped to its text and then fed to the LLM, which returns groups of text later used to construct ML sets.

#### Cannot-Link constraint set

Candidate points are uniformly randomly selected from the uncovered points with a distance greater than a certain threshold.
Then, the sampled points are evaluated by an LLM to check whether they should actually be placed in a CL set.

### Clustering with LLM-generated constraints

The clusters are first initialized by combining the hard ML constraints and the k-means++ algorithm.

#### Cluster initialization

The clusters are initialized by computing the the center of mass of the hard ML constraints and then using them as seeds.

#### ML and CL clustering with penalty

The soft ML constraints are used to try to merge the clusters, if the resulting cost is lower than the one of keeping them separate.
Regarding the CL constraints, a maximum-sum approach is used for k-means clustering.

### Experiments and evaluation

The datasets used are tweet, banking77, clinc (I/D) and GoEmo.
For the baseline, FSC (Viswanathan et al. 2024) is used.
The metrics used are accuracy, normalized mutual information, rand index and adjusted rand index.

## 5. Eliciting Human Preferences with Language Models

Paper: https://arxiv.org/pdf/2310.11589

**Takeaways**: users' preferences can be better identified by haveing a conversation with an LLM.

This paper shows that interactive, conversation-based methods (such as talking to an LLM) produces higher informative value, reduces human effort, surfaces unanticipated nuances and it is overall better than traditional methods based on passive learning techniques.

## 6. TO-GATE: Clarifying Questions and Summarizing Responses with Trajectory Optimization for Eliciting Human Preference

Paper: https://arxiv.org/pdf/2506.02827

**Takeaways**: ask clarifying questions by identifying how good/bad a question is.

LLMs can understand human preferences through multi-turn dialogue, by asking clarifying questions.
The proposed method enhances question generation (using trajectory optimization) by using a clarification resolver (to generate optimal questioning trajectories) and a summarizer (to ensure task-aligned final responses).
This method improve standard elicitation tasks by 9.32%.

The improvement is made possible by avoiding only using successful conversation as training data, but also comparing good and bad dialog trajectories and learning questions lead to better answers.

## 7. Summaries as Centroids for Interpretable and Scalable Text Clustering

Paper: https://arxiv.org/pdf/2502.09667

**Takeaways**: improve clustering algorithms by replacing a classical k-means step with a generated summary of the current clusters, improving the contextual meaning of clusters.

k-NLPmeans and k-LLMmeans are 2 proposed text-clustering variants of k-means that periodically replace numeric centroids with textual summaries.

### Summarization

The proposal is to periodically replace numerical centroid updates with summarization steps.
Instead of just averaging embeddings, after a fixed number of iterations, a textual prototype summarizes each cluster and the result is later embedded to obtain the centroid to use for assignments.
This approach ('summary-as-centroid') captures a richer contextual meaning, compared to the classical k-means implementation.
This also produces clusters that are more interpretable and more semantically coherent.

The 2 proposed algorithm differ only in the summarizer they use (classical NLP methods vs LLM-based ones).

### Interpretability and scalability

Each summarization is coincise and human-readable, allowing interpretability and easy debugging.

### Relation to existing LLM-based clustering

Other publications face 2 main issues:

- scalability, as the numnber of LLM calls grows with the dataset size
- opaque optimization, as a result of combining prompts, greedy merges and similarity thresholds without an explicit objective

This work addresses both issues.

### Embedding computation

Various embedding models have been tested, including: DistilBERT, e5-large, S-BERT, text-embedding-3-small.

### LLM usage

The LLM prompt to summarize a cluster is domain dependent.
For instance, for Bank77 dataset, they used something similar to "The following is a cluster of online banking questions. Write a single question that represents the cluster concisely."

## 8. LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering

Paper: https://arxiv.org/pdf/2511.15424

**Takeaways**: no embeddings required. LLM-only strategy (dual-prompt strategy) Linear complexity w.r.t. dataset size.

LLM-MemCluster leverages dynamic memory to instill state awareness and a dual-prompt stratedy to enable the model to reason about and determine the number of clusters:

- **dynamic memory**: 
a memory mechanism maintains a dynamic set of cluster labels, making the LLM a state-aware clustering agent that can create, merge and refine clusters to ensure global consistency

- **granularity control**: 
a dual-prompt strategy is used to find the best number of clusters.
One prompt encourages the LLM to make the known clusters into broader categories,
while the other one encourages the discovery of more fine-grained topic

Note that this approach completely removes the need of an encoder model, working with no embeddings.
Also, its complexity is linear w.r.t. the size of the dataset, completing the clustering of N instances in exactly N steps.

The datasets used are the same ones from ClutserLLM, and the evaluation metrics are ACC, NMI and ARI.
The results are compared against baselines leveraging different methods (traditional, embedding-based and LLM-based).

## 9. Asking Clarifying Questions for Preference Elicitation With Large Language Models

Paper: https://arxiv.org/pdf/2510.12015

**Takeaways**: fine-tune an LLM to learn the best questions to ask.

This paper proposes a method for preference elicitaiton by asking clarifying questions.
It's inspired by how diffusion models work:

- the **forward pass** (where noise is added) follows the process of removing profile elements that characterize the user's preferences.
After each piece of information is removed, an LLM is fine-tuned to ask a question that specifically targets and reveals the missing information
- in the **reverse process** (where noise is removes), the LLM generates questions based on what it learned in the previous step, and incrementally builds the user's profile (as it learned a **funneling strategy**)

## 10. Improving Text Embeddings with Large Language Models

Paper: https://aclanthology.org/2024.acl-long.642.pdf

**Takeaways**: generate good embeddings by taking into account user's preferences.

This paper proposed a method to create embeddings that are coherent with the context of the documents, 
and to ajust the embeddings so that they take into account the user's preferences (based on the instruction passed to the encoder),
The result of this paper is `e5-mistral-7b-instruct`, which can be easily adapted to other tasks without the need of fine-tuning it.

## 11. NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models

Paper: https://arxiv.org/pdf/2405.17428

**Takeaways**: generate good embeddings by taking into account user's preferences.

This paper introduces a method to create text embeddings by using decoder-only LLMs.
It achieves this by replacing causal masks with bidirectional attention, introducing a latent attention layer for vector pooling, and utilizing a two-stage instruction-tuning process.
It ranks 1st on the Massive Text Embedding Benchmark.


## 12. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models

Paper: https://arxiv.org/pdf/2506.05176

**Takeaways**: similar to NV-Embed but easier to use
