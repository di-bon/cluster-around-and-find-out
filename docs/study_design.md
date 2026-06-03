# Study design

## Factor structure

- **Independent variables (manipulated)**: during each run, the type of questions asked will be changed.
The ideal goal is to run 30 independent runs for each question type (specifically, yes/no questions, multiple choice questions and open-ended questions).
Additionally, the goal of the simulated user will change on each run.
The goal will repeat every 6 runs, so in the end there will be 5 runs per goal.
- **Dependent variables (observed)**: for each run, we will observe the turns to convergence, the tokens exchanged and the user response length.
The faithfulness of the proposed cluster is also measures, by using an LLM-as-judge.
More details about this process are available in the "Quality spec" document.
- **Constants**: during each run, the following parameters will be held constant:
dataset (a subset of the 500 questions from the StackOverflow full dataset), LLM model (qwen3.6 35B A3B), embedding model (qwen3 embedding 8B), number of initial clusters discovered using K-means (to enrich the system prompt of the interview agent), user persona of the simulated user.

### Outcomes

Using the metrics stated in "Quality spec", we define a primary outcome and some secondary outcomes.

- **Primary outcome**: turns to convergence.
This is the most direct measure of conversation efficiency and drives the main conclusion about which question type is best.
- **Secondary outcomes**: total tokens exchanged, user response tokens, and mean faithfulness score (average of coherence, alignment, and separation).
These are analyzed after the primary outcome and interpreted as supporting evidence, not as the basis for the headline claim.

Turns to convergence is designated the primary outcome for Q1 because it is objective and directly measures efficiency.
Faithfulness is treated as a secondary outcome despite being arguably more consequential, because it is measured via LLM-as-judge and therefore carries additional measurement noise.
A condition that wins on turns but loses on faithfulness would be a notable finding and is discussed explicitly in the decision rule.

### Decision rule

Decision rule:
> For each pairwise comparison between conditions, we compute the difference in means and a 95% bootstrap confidence interval (10,000 resamples).
> We conclude that condition A is more efficient than condition B if the 95% CI on the difference (A − B) excludes zero.

This rule allows us to compare two different question types and assess which one is better with a confidence of 95%.

## Baseline

In the baseline condition, documents are embedded without a task instruction.
In the experimental conditions, the task instruction is derived from the elicited user preferences.
This means the baseline differs from the experimental conditions in two ways: no preferences are elicited, and no instruction is provided to the embedding model.
The baseline is used primarly to try to answer Q2, as it results to be almost uninformative for Q1.

## Sample-size justification

The ideal number of runs is 30 per questions type, summing up to a total of 90 runs.
This is a tradeoff between the resources (API limits), time available and statistical robustness of the study.

## Limitations

As stated in the "Baseline" section, the baseline differs from the experimental conditions in two ways: no user preferences are elicited, and no task instruction is provided to the embedding model.
These two factors are confounded — the study cannot determine whether any observed improvement in faithfulness is driven by the preference elicitation, the instruction-guided embedding, or both.
Isolating these effects would require an additional condition (e.g. instruction-guided embedding without elicitation) which is left for future work.

Moreover, turns to convergence is designated the primary outcome for Q1 because it is objective and directly measures efficiency.
Faithfulness is treated as a secondary outcome despite being arguably more consequential, mainly because it refers to Q2 and also because it is measured via LLM-as-judge and therefore carries additional measurement noise.
This is also caused by time constraints, due to which the LLM-as-judge will not be a primary focus of development relative to other system modules.
Nevertheless, a condition that wins on turns but loses on faithfulness would be a notable finding and is discussed explicitly in the decision rule.

## Hypothesis testing

The only hypothesis that will be tested in this study is:

> Multiple choice questions are the best ones to be asked, as they represent the best tradeoff between user effort and information gained per turn.

## Experiments

To try to answer Q1 (thus, testing the hyphotesis mentioned above) and Q2, we will run N=108 different experiments.
Each question type (for a total of 3 quesitons) will be tested 36 times.
The simulated user will have a total of 18 different (persona, goal) combinations, meaning that each combination will be run 2 times.
The number of personas used is 3 and the number of goals is 6.
The exact personas and goals are available in the GitHub repository.

The evaluation of the results of the experiments will be performed as described above in this document. 
