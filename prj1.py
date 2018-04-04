# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:00:17 2017

@author: Qisen
"""

import random
import math
import matplotlib.pyplot as plt

def getSeq(S, S_start):
    seq = [S_start] # list seq to record episode
    cur = S_start # state pointer set to start 
    
    while True:
        action = random.choice([0,1])
        
        if action == 1:
            cur += 1
        if action == 0:
            cur -= 1
        seq.append(cur)    
        if cur == 0 or cur == len(S)-1:
            break;

    return seq

def getTrainSet(num_seq, S, S_start):
    train = []
    
    for seqNum in range(num_seq):
        train.append(getSeq(S, S_start))
        
    return train

def getDeltaWAfterSeq(seq, W, alpha, lmbd):
    
    deltaW = [0 for w in W]

    e_old = [0 for x in range(7)]
    for step, s in enumerate(seq[0:-1]):
        e = []
        X = [0 for x in range(7)]
        X[s] = 1

        w = W
        P_cur = sum([a*b for a, b in zip(w, X)])

        if seq[step+1] == 0:
            P_next = 0.0
        elif seq[step+1] == 6:
            P_next = 1.0
        else:
            X_next = [0 for x in range(7)]
            X_next[seq[step+1]] = 1
            P_next = sum([a*b for a, b in zip(w, X_next)])
            
        e = [lmbd*(e_oldt)+xt for e_oldt, xt in zip(e_old, X)]
        deltaW = [dw + (P_next - P_cur)*et for dw, et in zip(deltaW, e)]
        
        e_old = e

    return deltaW

        
def getDeltaWAfterSet(train, W, alpha, lmbd):
    deltaW = [0 for w in W]
    
    for seq in train:
        #print ('seq:',seq)
        deltaW_new = getDeltaWAfterSeq(seq, W, alpha, lmbd)
        deltaW = [x+y for x, y in zip(deltaW, deltaW_new)]

    return deltaW
    
def main():
    S = [x for x in range(7)] # numerically coded states 0-6
    S_start = 3 # starting state D=S[3]
    W_actual = [0, 1/6, 1/3, 1/2, 2/3, 5/6, 1] #actual probability in each state
    
    num_train = 100 # number of training sets
    num_seq = 10 # number of sequences per training set
    
    lambda_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    #lambda_list = [0.1]
    W_init = [0] + [0 for x in range(5)] + [1]
    alpha = 0.002
    maxNum_repeat = 10000
    thred_conv = 0.0001
    error_exp1 = {lmbd: 0 for lmbd in lambda_list}
    
    train_sets = []
    for trainIter in range(num_train):
        train = getTrainSet(num_seq, S, S_start)
        train_sets.append(train)

    ##============================ Experiment One    
    for lmbd in lambda_list:
        print('lambda:',lmbd)
        for train in train_sets:
            W = W_init
            # loop until convergence for each training set
            for repeat in range(maxNum_repeat):
                W_old = W
                deltaW = getDeltaWAfterSet(train, W, alpha, lmbd)
                #print ('deltaW:', deltaW)
                W = [x + alpha*y for x, y in zip(W, deltaW)]
                #print ('W:', W)
                diff = sum([abs(x-y) for x, y in zip(W, W_old)])
                if diff <= thred_conv:
                    #print ('W-found:',W)
                    break
                if repeat == maxNum_repeat:
                    print ('max repeat reached before convergence')
            # Calculate RMS error
            rms = math.sqrt(sum([pow((x-y),2) for x, y in zip(W_actual, W)])/5)
            error_exp1[lmbd] += rms
    
        # Average over RMS of each training set
        error_exp1[lmbd] = error_exp1[lmbd]/num_train
    
    print (error_exp1)

    plt.plot(list(error_exp1.keys()), list(error_exp1.values()))
    plt.ylabel('RMSE')
    plt.xlabel('Lambda')
    plt.show()

    ##========================== Experiment Two
    W_init = [0] + [0.5 for x in range(5)] + [1]    
    lambda_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    alpha_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, \
                  0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    error_exp2 = {lmbd: {alpha: 0 for alpha in alpha_list} for lmbd in lambda_list}

    for lmbd in lambda_list:
        for alpha in alpha_list:
            for train in train_sets:
                W = W_init
                deltaW = []
                
                for seq in train:
                    deltaW = getDeltaWAfterSeq(seq, W, alpha, lmbd)
                    W = [x + alpha*y for x, y in zip(W, deltaW)]
                rms = math.sqrt(sum([pow((x-y),2) for x, y in zip(W_actual, W)])/5)
                error_exp2[lmbd][alpha] += rms 

            error_exp2[lmbd][alpha] = error_exp2[lmbd][alpha]/num_train     
    print (error_exp2)
    
    fig2_lmbd_showList = [0, 0.3, 0.8, 1] 
    for lmbd in fig2_lmbd_showList:
        plt.plot(list(error_exp2[lmbd].keys()), list(error_exp2[lmbd].values()), \
                 label = str(lmbd))   
    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Alpha')
    plt.ylim([0, 1])
    plt.show()
    
    best_alpha = {x:0 for x in lambda_list}
    best_error = []
    for lmbd in lambda_list:
        best = min(error_exp2[lmbd], key=error_exp2[lmbd].get)
        best_alpha[lmbd] = best
        best_error.append(error_exp2[lmbd][best])
    print (best_alpha)
    
    plt.plot(lambda_list, best_error)
    plt.ylabel('RMSE using best alpha')
    plt.xlabel('Lambda')
    plt.show()
    
if __name__ == "__main__":
    main()
