# Minimax Group Fairness



We implement the `MinimaxFair` algorithm (which uses the famous Polynomial Weights algorithm) described in the paper here: https://arxiv.org/abs/2011.03108. This algorithm outputs a uniform distribution over models from a model class H, that minimise the maximum error of any defined group to epsilon-approximation of a Nash equilibrium.

Our analysis includes some extensions to the paper, namely:

- We define an adaptive group finding function (`G_` in `algos.py`), which finds groups that are most affected by a model in some model class H. This improves over the 'hand-drawn' group finding done in the source paper.
- In theory, the guarantees of the algorithm hold when we are solving a convex optimisation problem. However, most ML model/cost function pairs are non-convex problems. We explore whether the empirical guarantees continue to hold for non-convex cases.

You can find our in-depth analysis and results in the included `final_report.pdf`


This is done as final project for CIS 523: Ethical Algorithm Design. 

Team Members: Brian Williams, Campbell Phalen, Karan Sampath, Rohan Gupta


