import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as rfc
import sklearn.linear_model as sgd

f = open("training_data.txt") # 4189 data points total
g = open("testing_data.txt")
count = 0
x = []
y = []
val = ([],[])
test = []
tot = 4189
param = ""

# Read and format input
for l in f:
    count +=1; # Ignore first line
    if count ==1:
        l = l.strip()
        d = l.split('|')
        param = d
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
f.close()

# Testing data
count = 0
for l in g:
    count +=1; # Ignore first line
    if count ==1:
        continue
    d = l.split('|')
    d = map(lambda x:float(x),d)
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

#Collecting data t graph
msl_vals = []
s_error = []
v_error = []

#Setting model parameters
clf = sgd.SGDClassifier()
clf.set_params(penalty='l1')
clf.fit(x, y)

s_error.append(error_calc(clf, x, y))
v_error.append(error_calc(clf, val[0], val[1]))

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
print to_remove

plot_flag = False
if plot_flag == True:
    plt.figure(1)
#   plt.plot(msl_vals, s_error, label="Sample error")
#   plt.plot(msl_vals, v_error, label="Validation error")
    plt.legend(loc='best')
    plt.show()

print_flag = False
if print_flag == True:
    h = open("sgd_l1.txt", "w")
    h.write("Id,Prediction")
    for i in range(len(test)):
        h.write("\n")
        test_out = clf.predict(test[i])
        h.write(str(i+1) + ",")
        h.write(str(int(test_out[0])))
    h.close()
