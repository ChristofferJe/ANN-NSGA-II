# ANN-NSGA-II
Implementation of the Artificial Neural Network (ANN) NSGA-II algorithm. 

The classes in this repository is defined to implement the ANN-NSGA-II and NSGA-II algorithms as descriped in my report.
The `test.py` shows how to use both the ANN-NSGA-II and NSGA-II algorithm on the instance of the assignment problem with size 200. 

Below is a short description of the different classes used to implement the algorithms

1. class **NSGAII** defined in `NSGAII.py`: this is the class which implements the NSGA-II aglorithm, the methods in this class can be used to performe the operations of the NSGA-II algorithm on a **Population** class element. When combined with class **ANNSwap** defined in `ANNSwap.py` this becomes the ANN-NSGA-II algorithm.
2. class **ANNSwap** deinfed in `ANNSwap.py`: this is the class which defines the MLP used in the ANN-NSGA-II algorithm.
3. class **Evolution** defined in `evolution.py`: this is the class which uses a **NSGAII** class element to evolve a **Population** class element over multiple generations. This takes an argument `hybrid`, deciding whether to use the ANN-NSGA-II algorithm (`hybrid=True`) or the NSGA-II algorithm (`hybrid=False`). If hybrid=True then it combines  **NSGAII** with **ANNSwap** in the generate `generate_offspring` method from **NSGAII**.
