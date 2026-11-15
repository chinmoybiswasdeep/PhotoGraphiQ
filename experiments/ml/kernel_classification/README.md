# Classical data with the MuTA kernel

Install `.[ml-experiments]`, then run `python experiments/ml/kernel_classification/main.py`.
The JSON configuration specifies the 160/40 train/test split, seeds and SVM C.
Raw coordinates become angles without hidden scaling. Dataset generators are
fixed in the script. Held-out accuracy and confusion matrices are saved per
seed, alongside RBF-SVM and logistic-regression baselines. The figure displays
held-out predictions from the first configured seed; it is not a selected best run.

This studies the paper's Eq. 5 kernel on the same three dataset families, with
explicit local generator settings. It is not an exact reconstruction of Fig. 8.
MentPy agreement for the underlying logical map is covered by reference tests.
