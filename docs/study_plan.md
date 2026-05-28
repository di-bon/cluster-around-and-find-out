# Study Plan

v2 - 2026-05-28

## Changelog

### v2 (2026-05-28)

- Update probem statement
- Update research questions
- Add 'Provenance, ethics and exploration' section for the datasets used 

### v1 (2026-05-21) 

- Refine problem description 
- Focus the research questions on the conversational aspect of the system 
- Add datasets that will be used 
- Add simplifying assumptions

### v0 (2026-05-14) 

- Basic description of the problem 
- Propose some research questions

## Problem statement

Classical clustering requires the user to make a series of low-level technical decisions: 
choosing an algorithm (K-means, HDBSCAN, hierarchical clustering), selecting an embedding space, and tuning hyperparameters - all before seeing whether the result matches their intent.
This workflow is opaque and places the cognitive burden in the wrong place: the user must translate a high-level intent ("group these by topic") into technical choices they may not understand.

This project studies the viability of an alternative:
a conversational clustering system where the user expresses their preferences in natural language through a dialogue with an LLM.
The LLM elicits the user's intent through targeted questions, translates it into clustering constraints, and proposes a clustering assignment.
The user can then accept it or refine it  through further natural language feedback ("split cluster 1",  "merge clusters 2 and 5").

This project assumes that, given a good enough conversational clustering system that allows users to easily express their preference,
this can produce comparable or better results than a classical clustering approach (i.e. manually fine-tuning clustering algorithms, such as K-means).

## Questions

The following questions guide the project. 
**Q1 is the primary question** and the one the study is designed to answer. 
Q2 is a secondary question that might be addressed if time permits.

### Q1 (Primary) — Conversation efficiency
*Given a conversational clustering system, what properties of the conversation affect how quickly and faithfully the user's intent is captured? 
Specifically: which question type (open-ended / binary pairwise / example triplet) is the best to ask, in terms of conversation length (number of turns) and total tokens exchanged?*

### Q2 (Secondary) - Faithfulness
*How faithful is the proposed result, given the preferences of the user?
If the user requires additional changes (i.e. merging or splitting clusters), can the system update its representation to meet the user's preference?
If so, how many turns does this process take?*

### Other open questions

Other open questions that will not be addressed in this project, but hopefully will in future versions.
Check next paragraph "Simplifying assumptions" also, as some of these questions try to relax these assumptions.

- *Will the system be able to process large datasets (e.g. hundreds of thousands or more datapoints)?*
- *Can/does the system work on non-text data (tabular, image, time-series)? The current design assumes text input throughout*
- *How does the system behave when the user has no strong clustering preference and cannot answer the LLM's questions meaningfully?*
- *How robust is the system to vague, ambiguous, or deliberately unhelpful user answers?*
- *Can the system update the clustering as new data arrives, without restarting the conversation?*
- *Can the system explain why a specific datapoint was assigned to a specific cluster in terms the user understands?*

## Simplifying assumptions

Here are listed the simplifying assumptions for this project:

- **Responsive users**: when prompted, the user always provides a clear 
  answer expressing a preference or assigning a datapoint to a cluster. 
  Abstentions and "I don't know" responses are out of scope.
- **Consistent preferences**: the user's preferences are stable within a 
  session. They do not contradict earlier answers.
- **Text data only**: the system operates on text documents. Multimodal 
  data is out of scope.
- **Offline clustering**: the system produces a single clustering at the 
  end of the conversation, not incrementally during it. Then, it can be
  refined if the user wants to.

## Datasets

The datasets that will be used are:

- StackOverflow: 
it contains around 20k short texts (the questions' titles) and it allows users to express their clustering preferences based on different topics (e.g. programming language, debugging, architectural questions, etc.).
It also assigns each questions to a label (out of a total of 20 labels) to be used as ground-truth, but this information will not be used by this project.
The dataset is available [here](https://github.com/jacoxu/StackOverflow).

- GoEmotions:
it contains around 58k Reddit comments and it provides human annotations (28 classes in total, based on emotions) to be used as ground-truth.
The dataset is available [here](https://github.com/google-research/google-research/tree/master/goemotions).

### Provenance, ethics and exploration

- StackOverflow:
this is a public dataset, created by scraping StackOverflow's questions titles.
For this reason, the data should not contain any personal information, meaning that no data anonymization process is required.
No bias that could harm the results of this project was identified in this dataset.

- GoEmotions:
this is a public dataset, created by Google Research by extracting the corpus of Reddit comments.
Given the nature of the data, the authors attempted to remove sensitive aspects of it.
The content of the dataset, while potentially containing problematic content, should not influence the final results of this project.
More considerations about this dataset are available [here](https://github.com/google-research/google-research/blob/master/goemotions/goemotions_model_card.pdf).
