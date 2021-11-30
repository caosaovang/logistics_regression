import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

data = pd.read_csv('data_classification.csv',header=None).values
N, d = data.shape
x = data[:, 0:d - 1].reshape(-1, d - 1)
y = data[:, 2].reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)
X_train = data[:, 0:d - 1].reshape(-1, d - 1)
X_train = np.hstack((np.ones((N, 1)), X_train))
y_train = y = data[:, 2].reshape(-1, 1)

#for showing chart
true_x=[]
true_y=[]
false_x=[]
false_y=[]
for i in data :
    if i[2] == 1 :
        true_x.append(i[0])
        true_y.append(i[1])
    else :
        false_x.append(i[0])
        false_y.append(i[1])

#main code
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 
def predict(features,weights):
    z = np.dot(features,weights)
    return sigmoid(z)
def loss_function(features,labels,weights):
    prediction = predict(features,weights)
    loss_class1 = -labels*np.log(prediction)
    loss_class2 = -(1-labels)*np.log(1-prediction)
    loss = loss_class1+loss_class2
    return np.sum(loss)
def decisionBoundary(p) :
    if p >= 0.5 : return 1
    else: return 0
def update_weight (features,labels,weights,learning_rate) :
    n = len(labels)
    prediction = predict(features,weights)
    weights_temp = np.dot(features.T,(prediction-labels))/n
    updated_weight = weights-weights_temp*learning_rate
    return updated_weight
def train (features, labels, weights, learning_rate, iter) :
    history_loss=[]
    for i in range (iter):
        weights = update_weight(features,labels,weights,learning_rate)
        loss = loss_function(features,labels,weights)
        history_loss.append(loss)
        # show the loss function:
        # print("Loss in {} is : {}".format(i, history_loss[i]))
    return weights,history_loss


w = np.array([0., 0.1, 0.1]).reshape(-1, 1)
numOfIteration = 100000
cost = np.zeros((numOfIteration, 1))
learning_rate = 0.01
w,loss = train(X_train,y_train,w,learning_rate,numOfIteration)

# testing data
X_test = data[:, 0:d - 1].reshape(-1, d - 1)
X_test = np.hstack((np.ones((N, 1)), X_test))
y_test = data[:, 2].reshape(-1, 1)
l_test = len(X_test)
for i in range(l_test):
    temp = predict(X_test[i],w)
    if (decisionBoundary(temp)==1) :
        print(X_test[i], ': 1 :', y_test[i])
    else: print(X_test[i], ': 0 :', y_test[i])


x_check = [1,9,3]
temp= predict(x_check,w)
print("Value will be predicted for student who slept {} hours and studied {} hours".format(x_check[1],x_check[2]))
if (decisionBoundary(temp)==1) :
    print("Predict value is {}. So this student will pass!!".format(decisionBoundary(temp)))
else: print("Predict value is {}. So this student will fail!!".format(decisionBoundary(temp)))


# showing chart
plt.scatter(true_x,true_y,marker="o",c="y",edgecolors='none', s=30, label='Pass')
plt.scatter(false_x,false_y,marker="o",c="r",edgecolors='none', s=30, label='Fail')
plt.legend(loc=1)
plt.xlabel('Studied')
plt.ylabel('Slept')
plt.show()
yTime_series = np.array([i for i in range(numOfIteration)])
plt.plot(yTime_series,loss)
plt.xlabel("Time")
plt.ylabel("Loss")
plt.show()