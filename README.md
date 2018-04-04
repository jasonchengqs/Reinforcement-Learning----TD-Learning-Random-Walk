# CS7642 Reinforcement Learning (Project 1) -- Temporal Difference Algorithm & Random Walk
  File -- prj1.py (written in python 3)
  ## External packages included: 
    1) "random" package for generation of random sequences;
    2) "math" package for calculation of root mean square error;
    3) "matplotlib" for plotting.
  ## Function List:
    1) getSeq(S, S_start) --> return one random walk sequence (list) generated from given state list (S) and starting point (S_start)
    2) getTrainSet(num_seq, S, S_start) --> return one training set (list), which contains 100 random walk sequence (list)
    3) getDeltaWAfterSeq(seq, W, alpha, lmbd) --> return the sum of step-wise updates (list) of weight vector through the given random walk        sequence (seq).
    4) getDeltaWAfterSet(train, W, alpha, lmbd) --> return the sum of sequence-wise updates (list) of weight vector through the given       
       training set (train)
    5) main() --> main function contains definition of parameters, variables, and execution of experiments via calling all the helper 
       functions.
  ## Execution and printing results:
     Please runing the .py file directly, and the following results will be printed to the console:
    1) lambda: x --> showing the sweeping of lambda in experiment 1. 
    2) {0: 0.11323468597956379, 0.1: 0.11310629475316959, 0.3: 0.11353313615590302, 0.5: 0.11521753601397425, 0.7: 0.11973529652320351, 
        0.9: 0.13626137065437552, 1: 0.1668400047659492} --> a dictionary showing the errors of each lambda with format of {lambda: error}
    3) reproduced figure 3 in the paper
    4) a dictionary showing the errors of each lambda with different alpha. format: {lambda: {alpha: error}}
    5) reproduced figure 4 in the paper
    6) a dictionary showing the found best alpha corresponding to different lambda.
    7) reproduced figure 5 in the paper
