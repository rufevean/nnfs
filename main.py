"""
    mulilayered , feedforward, multiperceptron neuron 
"""

"""
inputs = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]] 
biases = [2,3,0.5]
"""
"""
a simple output function is summation of all multiplications of weights and inputs and adding it with bias 


output = lambda summ, inputs, weights: summ + sum(inputs[i] * weights[i] for i in range(len(inputs))) + bias ( SIMPLY A DOT PRODUCT OF WEIGHTS AND INPUTS and add it to BIASES) 

print(f"output : {output(summ,inputs,weights)}")

you can tweak weights and bias but not the inputs

"""

#    def mul(inputs,weights):
#        summ = 0
#        for i in range(len(inputs)):
#            summ += inputs[i]*weights[i]
#        return summ
#    def out(inputs,weights,bias):
#        return mul(inputs,weights) + bias
#

"""
def out(weights,biases,inputs,output):
    for weights,bias in zip(weights,biases):
        neuron_out = 0  
        for neuron_in ,weight in zip(inputs,weights):
            neuron_out = neuron_in * weight
        neuron_out = neuron_out + bias
        output.append(neuron_out)
    return output

if __name__ == '__main__':
    print(out(weights,biases,inputs,output))

=== 
A tensor is an object that can be represented as an array 
(NOT AN ARRAY)
"""


## P3 - The dot product
import numpy as np

"""

inputs = [[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]] 
biases = [2,3,0.5]
weights2 = [[0.1,-0.14,0.5],[-0.5,0.12,-0.33],[-0.44,0.73,-0.13]]
biases2 = [-1,2,-0.5] 
weights_transpose  = np.array(weights).T

layer1_output  = np.dot(inputs,weights_transpose) + biases
weights2_transpose = np.array(weights2).T
layer2_output = np.dot(layer1_output,weights2_transpose)+biases2


Weights always comes first 
WHY?  try matrix multiplying biases * weights 

print(f"layer 1 output :\n  {layer1_output}")
print(f"layer 2 output :\n  {layer2_output}")



np.random.seed(0)
X = [[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]]

class Layer_Dense: 
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases 


layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)


layer1.forward(X)
print(f"layer 1 output :\n {layer1.output}")

layer2.forward(layer1.output)
print(f"layer 2 output :\n {layer2.output}")



-   step function
-   Sigmoid function
-   Rectifed linear function (ReLU) (Fast) (Most popular for hidden layers)
    -   So close to linear but so powerful for non-linear inputs 

Why even an activation function ?

-  for non-linear inputs

"""

import nnfs
nnfs.init()

from nnfs.datasets import spiral_data

X,y  = spiral_data(100,3) 

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
#layer2 = Layer_Dense(5, 2)

activation1 = Activation_ReLU()


layer1.forward(X)
print(f"layer 1 output :\n {layer1.output}")
activation1.forward(layer1.output)
print(f"activation1 output :\n {activation1.output}") 


#layer2.forward(layer1.output)
#print(f"layer 2 output :\n {layer2.output}")
