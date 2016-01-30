import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as rfc

f = open("training_data.txt")
g = open("testing_data.txt")
count = 0
x = []
y = []

# Read and format input
for l in f:
    count +=1; # Ignore first line
    if count ==1:
        continue

    l = l.strip()
    d = l.split('|')
    d = map(lambda x:float(x),d)
    x.append(np.array(d[:-1])) # Saving x-vector 
    y.append(d[-1]) # Saving y-value

def error_calc(clf, x_vals, y_vals):
    count = 0
    wrong = 0
    for (i,x_i) in enumerate(x_vals):
        if (clf.predict(x_i) != y_vals[i]):
            wrong += 1
        count += 1
    return float(wrong)/count

for msl in range(8,20,2):
    print "Min-sample-leaves: " + str(msl)
    clf = rfc.RandomForestClassifier()
    clf.set_params(min_samples_leaf=msl)
    clf.fit(x, y)
    print error_calc(clf, x, y)

