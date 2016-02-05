import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as rfc
import project_io

training_data = "training_data.txt" # 4189 data points total
testing_data = "testing_data.txt"

####################    FILE IO    ##################
# Constants, and places to store the data for the actual analysis
FRAC_TRAIN = 0.667

x = []
y = []
val = ([],[])

#f = ([param1, param2, ...], [(x0,y0), (x1, y1), ...])
f = project_io.parse_train(training_data) 

param = f[0]
tot = len(f[1])
for (i, (a,b)) in enumerate(f[1]):
    if i < FRAC_TRAIN * tot:
        x.append(a)
        y.append(b)
    else:
        val[0].append(a)
        val[1].append(b)

# Testing data
test = project_io.parse_test(testing_data)

####################   HELPER FUNCTIONS    ##################
def error_calc(clf, x_vals, y_vals):
    count = 0
    wrong = 0
    for (i,x_i) in enumerate(x_vals):
        if (clf.predict(x_i) != y_vals[i]):
            wrong += 1
        count += 1
    return float(wrong)/count

####################   TRAINING       ##################

### SETUP ###
clf = rfc.RandomForestClassifier()
#Number of trees, estimators, trees, nest
clf.set_params(n_estimators=50)

msl_vals = []
s_error = []
v_error = []

### TRAINING ###
search_leaf = 1 # Search for best leaf size if set to 1
if search_leaf == 1:
    for msl in range(2,14,2): # Minimum sample leaf
        msl_vals.append(msl)
        print "Min-sample-leaves: " + str(msl)
        clf.set_params(min_samples_leaf=msl)
        clf.fit(x, y)
        s_error.append(error_calc(clf, x, y))
        v_error.append(error_calc(clf, val[0], val[1]))
elif search_leaf == 2:
    msl = 11 # Used as a default, maybe be incorrect?  50 iterations suggest 8
    msl_vals.append(msl)
    print "Min-sample-leaves: " + str(msl)
    clf.set_params(min_samples_leaf=msl)
    clf.fit(x, y)
    s_error.append(error_calc(clf, x, y))
    v_error.append(error_calc(clf, val[0], val[1]))

####################   Handling output   ##################
plot_flag = (search_leaf == 1)
print_to_file_flag = True

if plot_flag == True:
    plt.figure(1)
    plt.plot(msl_vals, s_error, label="Sample error")
    plt.plot(msl_vals, v_error, label="Validation error")
    plt.legend(loc='best')
    plt.show()

if print_to_file_flag == True:
    h = open("test.txt", "w")
    h.write("Id,Prediction")
    for i in range(len(test)):
        h.write("\n")
        test_out = clf.predict(test[i])
        h.write(str(i+1) + ",")
        h.write(str(int(test_out[0])))
    h.close()
