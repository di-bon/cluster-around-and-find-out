# Goal and Objective

The user wanted documents grouped according to this goal:
{user_preference}
 
The following clusters were produced ({n_clusters} clusters, {n_noise} noise docs):
{cluster_text}
 
Rate the clustering on these three dimensions:
 
- coherence (1-5): are documents within each cluster semantically similar to each other?
- alignment (1-5): do the clusters reflect the user's stated goal?
- separation (1-5): are the clusters clearly distinct from one another?
 
## Output format

Respond ONLY with this JSON structure:
{{
  "coherence": <int 1-5>,
  "alignment": <int 1-5>,
  "separation": <int 1-5>,
  "reasoning": "<one short paragraph explaining your scores>"
}}
