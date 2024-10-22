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

inputs = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]] 
biases = [2,3,0.5]


output  = np.dot(weights,inputs) + biases

""" Weights always comes first 
WHY?  try matrix multiplying biases * weights 
""" 

print(f"output : {output}")
