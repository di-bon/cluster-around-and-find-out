# Quality spec

This document defines the evaluation metrics used to compare the results of the proposed approach.

## Background

Evaluating the proposed clustering solution is a challenging task, as there is no real ground truth and the results are subjective to the users:
given the same dataset and the same clustering solution, one user might accept the clustering assignemnt and another one might require changes to that.

It is worth noticing that, even if clustering is a central part of this project, it is not what it is being evalated.
The evaluation focuses on the goodness of the conversation between the LLM and the user and on how the proposed solution fits the user's preferences.
This is different than assessing whether a clustering algorithm actually produces 'good' clusters compared to a ground truth (that in this project is not taken into account).

## Metrics

The metrics that will be evaluated are about the conversation quality.
To assess this, we measure:

1. efficiency: 
for each run, the turns to convergence and tokens exchanged are measured.
Lower turns/fewer tokens are preferred, as the system can be perceived as easier to use for humans, requiring less cognitive load to express the clustering preferences.

1. question clarity:
the number of tokens used by the user to generate an answer.
Fewer tokens are an indication of a clearer, better question.

1. faithfulness:
check whether the final result refelects the user preferences.
This metrics is composed of multiple sub-metrics (listed below), all rated using a Likert scale (from 1 to 5), using an LLM-as-judge.
The reliability of the LLM-as-judge is evaluated by having a set of human-evaluated examples (at least 30 ideally, time permitting) used to compute the Spearman correlation.
Specifically, the measured sub-metrics are:
    1. coherence: measures how semantically similar the documents of the same cluster are
    1. alignment: measures if the clusters reflect the user's stated goal
    1. separation: measures if the clusters are clearly different from one another
Finally, these submetrics are grouped together by computing the mean score of them.
