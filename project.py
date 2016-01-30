import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as rfc

f = open("training_data.txt") # 4189 data points total
g = open("testing_data.txt")
count = 0
x = []
y = []
val = ([],[])
test = []
tot = 4189

removed_indices=[]
#Read and format input
for l in f:
    count +=1; # Ignore first line
    if count ==1:
        l = l.strip()
        d = l.split('|')
        stuff = d
        #for i in range(len(d)): 
        #    if d[i].isdigit():
        #        removed_indices.append(i)
        continue

    l = l.strip()
    d = l.split('|')
    d = map(lambda x:float(x),d)

    # Removing removed_indices
    for i in range(len(removed_indices)-1, -1, -1):
        d.pop(removed_indices[i])

    if count < int(2.0*tot/3):
        x.append(np.array(d[:-1])) # Saving x-vector 
        y.append(d[-1]) # Saving y-value
    else:
        val[0].append(np.array(d[:-1])) 
        val[1].append(d[-1]) 
f.close()

print "Input parsed!  " + str(len(removed_indices)) + " indices ignored."
for i in removed_indices: 
    print stuff[i]

## Read and format input
#for l in f:
#    count +=1; # Ignore first line
#    if count ==1:
#        continue
#
#    l = l.strip()
#    d = l.split('|')
#    d = map(lambda x:float(x),d)
#    if count < int(2.0*tot/3):
#        x.append(np.array(d[:-1])) # Saving x-vector 
#        y.append(d[-1]) # Saving y-value
#    else:
#        val[0].append(np.array(d[:-1])) 
#        val[1].append(d[-1]) 
#f.close()

# Testing data
count = 0
for l in g:
    count +=1; # Ignore first line
    if count ==1:
        continue
    d = l.split('|')
    d = map(lambda x:float(x),d)

    # Removing removed_indices
    for i in range(len(removed_indices)-1, -1, -1):
        d.pop(removed_indices[i])

    test.append(np.array(d)) 
g.close()

def error_calc(clf, x_vals, y_vals):
    count = 0
    wrong = 0
    for (i,x_i) in enumerate(x_vals):
        if (clf.predict(x_i) != y_vals[i]):
            wrong += 1
        count += 1
    return float(wrong)/count

clf = rfc.RandomForestClassifier()
#Number of trees, estimators, trees, nest
clf.set_params(n_estimators=100)

msl_vals = []
s_error = []
v_error = []

search_leaf = 2
if search_leaf == 1:
    for msl in range(8,16,1): # Minimum sample leaf
        msl_vals.append(msl)
        print "Min-sample-leaves: " + str(msl)
        clf.set_params(min_samples_leaf=msl)
        clf.fit(x, y)
        s_error.append(error_calc(clf, x, y))
        v_error.append(error_calc(clf, val[0], val[1]))
elif search_leaf == 2:
    msl = 11
    msl_vals.append(msl)
    print "Min-sample-leaves: " + str(msl)
    clf.set_params(min_samples_leaf=msl)
    clf.fit(x, y)
    s_error.append(error_calc(clf, x, y))
    v_error.append(error_calc(clf, val[0], val[1]))

plot_flag = (search_leaf == 1)
if plot_flag == True:
    plt.figure(1)
    plt.plot(msl_vals, s_error, label="Sample error")
    plt.plot(msl_vals, v_error, label="Validation error")
    plt.legend(loc='best')
    plt.show()

print_flag = True
if print_flag == True:
    h = open("test.txt", "w")
    h.write("Id,Prediction")
    for i in range(len(test)):
        h.write("\n")
        test_out = clf.predict(test[i])
        h.write(str(i+1) + ",")
        h.write(str(int(test_out[0])))
    h.close()
