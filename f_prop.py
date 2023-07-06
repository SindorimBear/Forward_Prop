import math
import numpy as np
#Basic forward propagation algorithm based on a blog
#Web Link: https://theneuralblog.com/forward-pass-backpropagation-example/

#Equation of sigmoid
def sigmoid(a):
    return np.exp(-np.logaddexp(0, -a))
#Equation of Error Percentage Calculation
def error(x,y):
    return (((1/2)*(y-x)**2))
#Receive input values of input 1 & 2, hidden layer 1 & 2's biases b1 & b2, and the expected output for each input e_out1 & e_out2
i1, i2, b1, b2, e_out1, e_out2 = map(float,input().split())
#Input the weight of each nodes
w1,w2,w3,w4,w5,w6,w7,w8 = map(float,input().split())

#Calculate up to the first hidden layer
h1 = i1 * w1 + i2 * w3 + b1
h2 = i1 * w2 + i2 * w4 + b1
h1 = sigmoid(h1)
h2 = sigmoid(h2)
#Calculate to the output layer
o1 = h1 * w5 + h2 * w6 + b2
o2 = h1 * w7 + h2 * w8 + b2
o1 = sigmoid(o1)
o2 = sigmoid(o2)

#Calculate the error percentage
e1 = error(o1, e_out1)
e2 = error(o2, e_out2)
e_total = e1 + e2

#Check if the value of each value is correct
"""
print(h1)
print(h2)
print(o1)
print(o2)
print(e1)
print(e2)
"""

#Print the total error percentage of the train set
print("the error percentage of input 1 is " + str(round(e_total, 5)))