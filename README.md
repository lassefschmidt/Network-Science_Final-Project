# Machine Learning in Network Science
Group project in Machine Learning for Network Science Class at CentraleSupelec

## About the Project

This repository contains our work during the final project within the Machine Learning in Network Science class at CentraleSupélec. We chose to continue working on a link prediction task (as in previous Kaggle challenge, our repository accessible via this [link](https://github.com/lassefschmidt/Network-Science_Challenge)). This time we will however apply what we learned on the CORA data set. One notable difference to the previous Kaggle challenge is that the CORA data set represents a directed graph while the Kaggle challenge covered link prediction on undirected graphs.

## About the data set
The CORA data contains bibliographic records of papers. It was originally published by Andrew McCallum in 2000 and since then various versions of the data set have been used. In this project, we will be using the version provided by Sen et al in 2008 (accessible via this [link](https://linqs.org/datasets/#cora)), which consists of 2,708 scientific publications from the domain of Machine Learning and in which each publication is classified into one of the following seven classes:
1. case based
2. genetic algorithms
3. neural networks
4. probabilistic methods
5. reinforcement learning
6. rule learning
7. theory

In addition, each publication in the data set is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary itself contains 1,433 unique words and is the result of the following preprocessing steps: (1) stemming and removing stopwords, (2) remove all words with document frequency less than 10.

Lastly, the network contains 5,429 links, where each link corresponds to a citation (target node -> source node). The papers were selected in a way such that in the final corpus every paper cites or is cited by at least one other paper.

To our knowledge no official train test split exists for the CORA data set and we will build our test set by randomly removing edges from the original graph and adding the same amount of none-edges.

## References
1. P. Sen, G. M. Namata, M. Bilgic, L. Getoor, B. Gallagher, and T. Eliassi-Rad,
“Collective Classification in Network Data,” AAAI, 29(3), 93-106 (2008).
2. E. C. Mutlu, T. Oghaz, A. Rajabi, and I. Garibay, “Review on learning
and extracting graph features for link prediction,” Mach. Learn. Knowl.
Extr. 2, 672–704 (2020).
3. S. J. Ahn and M. Kim, “Variational graph normalized AutoEncoders,” in
Proceedings of the 30th ACM International Conference on Information
&amp Knowledge Management, (ACM, 2021).
4. Y.-L. Lee and T. Zhou, “Collaborative filtering approach to link predic-
tion,” Phys. A: Stat. Mech. its Appl. 578, 126107 (2021).
5. A. Papadimitriou, P. Symeonidis, and Y. Manolopoulos, “Fast and
accurate link prediction in social networking systems,” J. Syst. Softw.85, 2119–2132 (2012). Selected papers from the 2011 Joint Working
IEEE/IFIP Conference on Software Architecture (WICSA 2011).
6. A. Grover and J. Leskovec, “node2vec: Scalable feature learning for
networks,” CoRR. abs/1607.00653 (2016)
