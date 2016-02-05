import numpy as np

# Moving input parsing to a different file

# This function assumes that we have a y-value at the end and 
# a starting line of parameters
# input: file name, portion of dataset used for training
# output: list of parameters, list of (x,y) pairs from dataset
# x is a numpy array, y is a number
def parse_train(filename, tot=4189):
    f = open(filename)
    train = []
    
    #Read and format input
    count = 0
    for l in f:
        count +=1;
        if count ==1: # Deal with first line of parameters
            l = l.strip()
            d = l.split('|')
            param = d
            continue
    
        l = l.strip()
        d = l.split('|')
        d = map(lambda x:float(x),d)
    
        train.append(  (np.array(d[:-1]), d[-1])  ) 
    f.close()
    return (param, train)

# Parses test data, no y value, no parameters list, just mapping with np.array
def parse_test(filename):
    g = open(filename)
    test = []
    count = 0
    for l in g:
        count +=1; # Ignore first line
        if count ==1:
            continue
        d = l.split('|')
        d = map(lambda x:float(x),d)
    
        test.append(np.array(d)) 
    g.close()
    return test
