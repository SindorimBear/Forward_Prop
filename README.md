# Forward_Prop

## Simple Forward Propagation Code

The purpose of this project is to show a basic structure of the Forward Propagation mentioned within in the web. Link: https://theneuralblog.com/forward-pass-backpropagation-example/

### Notice

In the blog, it mentions two equations required in the forward propagation.
The first being Sigmoid Equation:
-Basic purpose of sigmoid equation is to add non-linearity in the machine learning model
-Equation being: 1 / (1 + e ^ -x)

The second is error percentage:
-Purpose of this equation is to evaluate the error between the calculated output and the expected output

### Explanation of Code

The libraries required for this code is pretty simple. Math and Numpy
```
import math
import numpy as np
```

We make a def structure to avoid unncessary repetition of codes
First def structure defines the sigmoid equation.
```
def sigmoid(a):
    return np.exp(-np.logaddexp(0, -a))
```
Second def structure defines the error equation
```
def error(x,y):
    return (((1/2)*(y-x)**2))
```

Now we can start receiving the necessary values for forward propagation
- 2 input values: i1 & i2
- 2 hidden layer biases b1 & b2
- 2 expected output values: o1 & o2
```
i1, i2, b1, b2, e_out1, e_out2 = map(float,input().split())
```
A space between the input will separate the input values respectively

Next we need the necessary weight values for each nodes
-w1 ~ w8
```
w1,w2,w3,w4,w5,w6,w7,w8 = map(float,input().split())
```

After that we can find the hidden layer values for the first time:
The basic function in this equation is: 
1 / (1 + e ^ (-x))
Where x being each node value
Ex) h1 = i1 * w1 + i2 * w2 + b1

**Since each node has different biases, it is necessary to check your calculation mid-way**

After the calculation of each hidden layers are complete, we can calculate output values. The calculation of output values is same with the calculation of the hidden layers.
Ex) o1 = h1 * w5 + h2 * h6 + b2

Now that the expected values are out, we need to calculate the error between the expected output and the actual output.

Now sum the error percentage to get the total error
```
e1 = error(o1, e_out1)
e2 = error(o2, e_out2)
e_total = e1 + e2
```
