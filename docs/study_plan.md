# Study Plan

v1 - 2026-05-21

## Changelog

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

The central claim this project investigates is:

> Conversational clustering allows users to easily express their preferences,
> and at the same time achieving comparable or better results than using classical clustering algorithms (manually fine-tuned).

## Questions

The following questions guide the project. 
**Q1 is the primary question** and the one the study is designed to answer. 
Q2 is a secondary question that might be addressed if time permits.

### Q1 (Primary) — Viability and usability vs. classical clustering
*Does a conversational clustering system allow non-expert users to produce cluster assignments of comparable quality to those produced by expert users of classical clustering tools 
(e.g. K-means with manual hyperparameter tuning), while requiring less domain-specific technical knowledge?*

### Q2 (Secondary) — Conversation efficiency
*Given that the conversational interface works, what properties of the conversation affect how quickly and faithfully the user's intent is captured? 
Specifically: does question type (open-ended / binary pairwise / example triplet), conversation length (number of turns), and total tokens exchanged predict preference-alignment score?*

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
  end of the conversation, not incrementally during it.

## Datasets

The datasets that will be used are:

- StackOverflow: 
it contains around 20k short texts (the questions' titles) and it allows users to express their clustering preferences based on different topics (e.g. programming language, debugging, architectural questions, etc.).
It also assigns each questions to a label (out of a total of 20 labels) to be used as ground-truth, but this information will not be used by this project.
The dataset is available [here](https://github.com/jacoxu/StackOverflow).

- GoEmotions:
it contains around 58k Reddit comments and it provides human annotations (28 classes in total, based on emotions) to be used as ground-truth.
The dataset is available [here](https://github.com/google-research/google-research/tree/master/goemotions).
