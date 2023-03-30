# Machine Learning in Network Science
Group project in Machine Learning for Network Science Class at CentraleSupelec

## About the Project

This repository contains our work during the final project within the Machine Learning in Network Science class at CentraleSupélec. We chose to continue working on a link prediction task (as in previous Kaggle challenge, our repository accessible via this [link](https://github.com/lassefschmidt/Network-Science_Challenge)). This time we will however apply what we learned on the CORA dataset. One notable difference to the previous Kaggle challenge is that the CORA dataset represents a directed graph while the Kaggle challenge covered link prediction on undirected graphs.

## About the Dataset
The CORA dataset consists of 2,708 scientific publications from the domain of Machine Learning and each publication is classified into one of the following seven classes: (1) case based, (2) genetic algorithms, (3) neural networks, (4) probabilistic methods, (5) reinforcement learning, (6) rule learning, and (7) theory. Overall, the network contains 5,429 links, where each link corresponds to a citation (target node -> source node). The papers were selected in a way such that in the final corpus every paper cites or is cited by at least one other paper. In addition, each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary itself contains 1,433 unique words and is the result of the following preprocessing steps: (1) stemming and removing stopwords, (2) remove all words with document frequency less than 10.

The dataset was accessed via this [link](https://web.archive.org/web/20151007064508/http://linqs.cs.umd.edu/projects/projects/lbc/). To our knowledge no official train test split exists for the CORA dataset and we will build our test set by randomly removing edges from the original graph.
