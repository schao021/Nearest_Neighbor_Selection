# Feature Selection with Nearest Neighbor Classifier

## Introduction
This project focuses on implementing a nearest neighbor classifier within a wrapper method to perform feature selection. The core objective is to identify the most relevant features in a dataset to improve classification accuracy and reduce comuptational complexity. Two popular approaches are evaluated: forward selecrtion and backward elimination.

## Forward Selection vs Backward Elimination
### Forward Selection
This method begins with an empty feature set and adds one feature at a time, selecting the one that improves the model's performance the most at each step. The process continues until no further improvement is observed. This greedy strategy is effective in detecting strong individual features and avoids early inclusion of noisy data.
### Backward Elimination
Conversely, this method starts with all available features and removes one at a time, discarding the feature whose removal leads to the least drop (or greatest gain) in performance. It continues until further removal deteriorates accuracy. This technique is more aggressive and can help eliminate redundant or irrelevant features.
