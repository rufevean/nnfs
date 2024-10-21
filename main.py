"""
    mulilayered , feedforward, multiperceptron neuron 
"""

inputs = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]] 
bias = [2,3,0.5]

"""
a simple output function is summation of all multiplications of weights and inputs and adding it with bias 


output = lambda summ, inputs, weights: summ + sum(inputs[i] * weights[i] for i in range(len(inputs))) + bias 

print(f"output : {output(summ,inputs,weights)}")

you can tweak weights and bias but not the inputs

""" 

output = [] 

def mul(inputs,weights):
    summ = 0 
    for i in range(len(inputs)):
        summ += inputs[i]*weights[i]
    return summ 
def out(inputs,weights,bias):
    return mul(inputs,weights) + bias 

if __name__ == '__main__':
    for i in range(len(weights)):
        output.append(out(inputs,weights[i],bias[i]))
    print(f"output vector : {output}")


