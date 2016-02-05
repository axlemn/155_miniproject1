import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as rfc
import sklearn.linear_model as sgd
import project_io

####################    FILE IO    ##################
# Constants, and places to store the data for the actual analysis
FRAC_TRAIN = 2.0/3
training_data = "training_data.txt"

x = []
y = []
val = ([],[])

#f = ([param1, param2, ...], [(x0,y0), (x1, y1), ...])
f = project_io.parse_train(training_data) 

param = f[0]
tot = len(f[1])
for (i, (a,b)) in enumerate(f[1]):
    if i < int(FRAC_TRAIN * tot):
        x.append(a)
        y.append(b)
    else:
        val[0].append(a)
        val[1].append(b)

########## TRAINING ############
#Setting model parameters
clf = sgd.SGDClassifier()
clf.set_params(penalty='l1')
clf.set_params()
#clf.set_params(n_iter=1000)
clf.fit(x, y)

print clf.get_params()
print "Num variables used:" + str(len(clf.coef_[0]))

count = 0
to_remove = []
handle = clf.coef_[0]
for i in range(len(handle)):
    if abs(handle[i]) < .0001:
        count += 1
        to_remove.append([i,param[i],handle[i]])
print "suggested to remove:" + str(count)
#print to_remove
for r in to_remove:
    print r
