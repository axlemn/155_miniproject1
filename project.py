import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as rfc

f = open("training_data.txt") # 4189 data points total
#g = open("testing_data.txt")
count = 0
x = []
y = []
val = ([],[])
tot = 4189

# Read and format input
for l in f:
    count +=1; # Ignore first line
    if count ==1:
        continue

    l = l.strip()
    d = l.split('|')
    d = map(lambda x:float(x),d)
    if count < int(2.0*tot/3):
        x.append(np.array(d[:-1])) # Saving x-vector 
        y.append(d[-1]) # Saving y-value
    else:
        val[0].append(np.array(d[:-1])) 
        val[1].append(d[-1]) 

def error_calc(clf, x_vals, y_vals):
    count = 0
    wrong = 0
    for (i,x_i) in enumerate(x_vals):
        if (clf.predict(x_i) != y_vals[i]):
            wrong += 1
        count += 1
    return float(wrong)/count

clf = rfc.RandomForestClassifier()
clf.set_params(n_estimators=100)

for msl in range(8,16,2):
    print "Min-sample-leaves: " + str(msl)
    clf.set_params(min_samples_leaf=msl)
    clf.fit(x, y)
    print "Sample error: " + str(error_calc(clf, x, y))
    print "Validation error: " +  str(error_calc(clf, val[0], val[1]))


