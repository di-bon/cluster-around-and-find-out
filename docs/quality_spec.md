# Quality spec

This document defines the evaluation metrics used to compare the results of the proposed approach.

## Background

Evaluating the proposed clustering solution is a challenging task, as there is no real ground truth and the results are subjective to the users:
given the same dataset and the same clustering solution, one user might accept the clustering assignemnt and another one might require changes to that.

It is worth noticing that, even if clustering is a central part of this project, it is not what it is being evalated.
The evaluation focuses on the goodness of the conversation between the LLM and the user and on how the proposed solution fits the user's preferences.
This is different than assessing whether a clustering algorithm actually produces 'good' clusters compared to a ground truth (that in this project is not taken into account).

## Metrics

The metrics that will be evaluated are:

1. clustering quality:
the solhouette score is computed on the identified clusters.
This metric is only used as a sanity check, ensuring that no edge cases are produced (i.e. one cluster for each datapoint, or just a single cluster is produced as the result).

1. Conversation quality:
this is the core of this study.
It it composed of multiple metrics:

    1. efficiency: 
    for each run, the turns to convergence and tokens exchanged are measured.
    Lower turns/fewer tokens are preferred, as the system can be perceived as easier to use for humans, requiring less cognitive load to express the clustering preferences.

    1. question clarity:
    the number of tokens used by the user to generate an answer.

    1. faithfulness:
    check whether the final result refelects the user preferences.
    This is measured using a Likert scale (from 1 to 5), using an LLM-as-judge.
