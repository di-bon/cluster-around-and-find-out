# Quality spec

This document defines the evaluation metrics used to compare the results of the proposed approach.

## Evaluating clusters

Evaluating the proposed clustering solution is a challenging task, as there is no real ground truth and the results are subjective to the users:
given the same dataset and the same clustering solution, one user might accept the clustering assignemnt and another one might require changes to that.

While this issue reamins unsolved for now, the evaluation of an assignment can still be measured using classical metrics, such as the silouhette coefficient.
These methods can be applied to traditional clustering algorithms, but if the users modifies the output (e.g. by merging 2 clusters), then the results may not reflect what the user perceives in terms of quality.
To address this problem, the evaluation can be performed by not totally relying on user-modified clusters:
for instance, if clusters 1 and 2 are merged into the same one (cluster 3), then computing the silouhette coeffiecient on cluster 3 might produce a low score.
It may be better to compute the score of cluster 3 by computing the score of cluster 1 and 2 individually, and then use them to compute the score of cluster 3 by calculating the weighted average of clusters 1 and 2.

To mimic user's preferences, the LLM-as-a-judge method can be used (which is a fine-tuned LLM evaluator).
