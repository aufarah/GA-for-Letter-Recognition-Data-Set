import struct
import copy
import numpy as np
import matplotlib
from matplotlib import pyplot
import pandas as pd
from pandas import *

"""
    O 
               O 
    O                      O
               O 
    O                      O 
               O 
    O                      O
               O 
    O 
Layer 0     Layer 1     Layer 2
/Input      /Hidden     /Output

 # Neuron
 \
  \
   [Σ]-- z --[f]-- a
  /
 /
 
 z = Sigma of all weighted input (a from left layer/ a[l-1])
 a = activated z
 
 # Notation
    ----------------
    z[l][k], a[l][k]
    ----------------
 z = Sums
 a = activated sums
 l = layer number
 k = neuron number
 //index started at 0
 
    ---------------
    w[l][k,j],
    b[l][k]
    ---------------
 w = weight
 b = bias
 k = output layer (where weighted input summed placed)
 j = input layer
 //index started at 0
 
    O ------ z[0][0]
     \------ w[1][0,0]
      O ---- z[1][0]
     /------ w[1][0,1]
    O ------ z[0][1]
    
All operation is done in matrix with Numpy:
 #Matrix mode:
    - a,z, and b:
      a[l].T = [a[l][0], a[l][1], ... a[l][k]]
    - weight
      w[l] = [ w[l][0,0]  w[l][0,1]  ...  w[l][0,j] ]
             [ w[l][1,0]  w[l][1,1]  ...  w[l][1,j] ]
             [   ...         ...     ...     ...    ]
             [ w[l][k,0]  w[l][k,1]  ...  w[l][k,j] ]
    - Operation:
      z[l] = w[l] * a[l-1] + b[l]
      a[l] = f(z[l])    ... f = activation function
      Cost = e.T * e
           = e . e 
      e = output - t
      output = a[L] ... final layer (L) neurons

This training done by dividing datasets into several packages called
minibatches. Natural selection in each minibatch is called an age, and
after that there'll be chosen a best individu (chosen one),
if there're 8 minibatches, there'll be 8 ages and 8 chosen ones.

Then, the 8 chosen ones will join an final natural selection (afterlife)
to choose final best of the best. The superchosenone the saved in npz file.

Note that minibatch criterions must be the same as like in
gradient descent: elements in each minibatch must be representative
"""

numChild = 20
numGeneration = 300
initialpop = 40

def havesex(population):
    for i in range(numChild):
        #choosing 1st parent
        p1 = np.random.randint(len(population))

        #make child template
        child = copy.deepcopy(population[p1])

        #averaging parents (kumpul gebouw 5 persons)
        """
        for i in range(1):
            superweight = np.random.uniform(-1,1)
            person = np.random.randint(len(population))
            child.weights1 += superweight * population[person].weights1
            child.weights2 += superweight * population[person].weights2
            child.bias1    += superweight * population[person].bias1
            child.bias2    += superweight * population[person].bias2"""

        for i in range(4):
            child.weights1 += np.random.uniform(-1,1) * population[np.random.randint(len(population))].weights1
            child.weights2 += np.random.uniform(-1,1) * population[np.random.randint(len(population))].weights2
            child.bias1    += np.random.uniform(-1,1) * population[np.random.randint(len(population))].bias1
            child.bias2    += np.random.uniform(-1,1) * population[np.random.randint(len(population))].bias2

        
        child.weights1 /= np.random.randint(5) + 1
        child.weights2 /= np.random.randint(5) + 1
        child.bias1 /= np.random.randint(5) + 1
        child.bias2 /= np.random.randint(5) + 1

        """
        child.weights1 /= 5
        child.weights2 /= 5
        child.bias1 /= 5
        child.bias2 /= 5"""

        population  = np.append(population,child)
    return population

def findfit(population):
    result = np.array([])
    loops  = len(population)

    for i in range(loops):
        #feed forwarding all sample
        for j in range(0, y.shape[0]):
            population[i].feedind(j)

        #finding average cost of all sample
        cost = population[i].costtraining()
        result = np.append(result,cost)
        
    return result
    
def sort(input,output):
    for i in range(len(output)):
        lowIndex = i
        for j in range(i+1,len(output)):
            if output[j] < output[lowIndex]:
                lowIndex = j

        output[i],output[lowIndex] = output[lowIndex],output[i]
        input[i],input[lowIndex] = input[lowIndex],input[i]
    return [input,output]

def mutation(individu):
    individu.randominit()
    
class NeuralNetwork:
    def costtraining(self):
        #finding average cost from all training inputs
        cost = 0
        for i in range(0, self.y.shape[0]):
            errind = self.y[i] - self.output[i] 
            errind = np.array(errind)
            costind = np.dot(errind,errind)
            cost = cost + costind
        
        cost = cost / self.y.shape[0]
        return cost
    

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        #trivial : the inputs are activated neurons (to make equation simpler)
        #  a[l] = f(z[l])
        #   therefore argument for f'() must be z[l]
        #   but in this function the argument is a[l], a.k.a f'(a[l])
           
        return x * (1 - x)

    def relu(self, x):
        return x*(x>=0)
        
    def drelu(self, y):
        #computing derivative to the Sigmoid function
        #trivial : the inputs are activated neurons (to make equation simpler)
        #   y = f(z) = max(z)
        #       f'(z) = f'(y)
        #   therefore argument for f'() must be z[l]
        #   but in this function the argument is a[l], a.k.a f'(a[l])
        return 1.*(y>=0)

    def __init__(self, x, y):
        self.h1         = np.array([np.zeros(12)]).T             #hidden layer 1, 12 neuron. a[1] .
        self.input      = x                                     #. a[0] .           
        self.y          = y                                     #. t .
        
        self.randominit()
        #self.nguyeninit()
        #self.smartinit()
        
        # eta adalah learning rate
        self.eta        = 1
        self.output     = np.zeros(y.shape)
        self.error      = self.y - self.output

    def randominit(self):
        wates = 10
        #self.weights2   = np.random.random( (self.y.shape[1], self.h1.shape[0]) )      #horizontal per input m, verticaled as y n
        #self.bias2      = np.array([np.random.random(self.y.shape[1])]).T
        self.weights2   = np.random.uniform(-wates,wates,[self.y.shape[1], self.h1.shape[0]])
        self.bias2      = np.random.uniform(-wates,wates,[self.y.shape[1],1])

        #self.bias1      = np.array([np.random.random(self.h1.shape[0])]).T
        #self.weights1   = np.random.random( (self.h1.shape[0], self.input.shape[1]) )
        self.weights1   = np.random.uniform(-wates,wates,[self.h1.shape[0], self.input.shape[1]])
        self.bias1      = np.random.uniform(-wates,wates,[self.h1.shape[0],1])


    def smartinit(self):
        read = np.load('letterNBSS.npz')
        self.weights2   = read['w2']
        self.bias2      = read['b2']

        self.bias1      = read['b1']
        self.weights1   = read['w1']

    def smartinitload(self,file):
        read = np.load(file)
        self.weights2   = read['w2']
        self.bias2      = read['b2']

        self.bias1      = read['b1']
        self.weights1   = read['w1']

    def nguyeninit(self):
        """"
        [w00 w01] == magnitude ==> sqrt( [w00 w01] . [w00 w01] ) 
        [w10 w11] == magnitude ==> sqrt( [w10 w11] . [w10 w11] )

        Update:
        w[i][..] = beta * w[i][..] / magnitude[i]
        
        """
        self.randominit()
        
        beta        = 0.7*(self.input.shape[1]**(1/self.h1.shape[0]))
        self.bias1  = np.random.uniform(-beta,beta,[self.h1.shape[0],1])
        for i in range(self.weights1.shape[0]):
            magn = np.sqrt(np.dot(self.weights1[i],self.weights1[i]))
            self.weights1[i] = beta * self.weights1[i] / magn
            
        beta        = 0.7*(self.h1.shape[0]**(1/self.y.shape[1]))
        self.bias2  = np.random.uniform(-beta,beta,[self.y.shape[1],1])
        for i in range(self.weights2.shape[0]):
            magn = np.sqrt(np.dot(self.weights2[i],self.weights2[i]))
            self.weights2[i] = beta * self.weights2[i] / magn    
        
        
    def feedall(self):
        #feedforward all training couples
        for j in range(0, y.shape[0]):
            self.feedind(j)
            
    def feedind(self,n):
        #feedforward function for a training couple
        z1 = np.matrix(self.weights1) * np.matrix(self.input[n]).T + self.bias1 
        z1 = np.array(z1)
        a1 = self.sigmoid(z1)
        self.h1 = a1

        z2 = np.matrix(self.weights2) * np.matrix(a1) + self.bias2
        z2   = np.array(z2).T
        a2   = np.array(self.sigmoid(z2))
        self.output[n] = a2
    
    def test(self,x):
        #feedforward function for testing
        #index input = 0, hidden = 1, output = 2
        #z = sum, a = activated
        z1 = np.matrix(self.weights1) * np.matrix(x).T + self.bias1
        z1 = np.array(z1)
        a1 = self.sigmoid(z1)

        z2 = np.matrix(self.weights2) * np.matrix(a1) + self.bias2
        z2   = np.array(z2).T
        a2   = np.array(self.sigmoid(z2))

        return a2

def geneticStart():
    global population,newpop,fitness
    for c in range(numGeneration):
        #mating
        population = havesex(population)

        #find fitness of populations
        fitness = findfit(population)
        #fitness = stochfindfit(population)

        #sorting
        sort(population,fitness)

        #kill the unfits (died == numchild)
        initial = len(population)
        for i in range(numChild):
            index = i + 1
            population = np.delete(population,initial-index)
            fitness   = np.delete(fitness,initial-index)
        print(c,fitness[0])

        if (fitness[0] <= 1e-12):
            break


if __name__ == "__main__":

    #training couples
    df = pd.read_excel('letterBinaryNorm.xlsx', header=None)
    npa= np.array(df)
    sy = npa[0:36,0:3]
    sx = npa[0:36,3:19]

    minibatches = int(sy.shape[0]/3)
    chosenpop = np.array([])
    for i in range(minibatches):
        indeks = i*3
        x = npa[indeks:indeks+3,3:19]
        y = npa[indeks:indeks+3,0:3]
        #creating initial population / ancestors
        population = np.array([])
        for i in range(initialpop):
            population = np.append(population, NeuralNetwork(x,y))
        geneticStart()
        chosenpop = np.append(chosenpop,population[0])
    
    population = chosenpop
    x = sx
    y = sy
    numchild = int(population.shape[0])
    for i in range(population.shape[0]):
        population[i].input = x
        population[i].y = y
        population[i].output = np.zeros(y.shape)
    print(population.shape[0])
    geneticStart()
    np.savez('letterNBSSga.npz', w1=population[0].weights1, b1=population[0].bias1, w2=population[0].weights2, b2=population[0].bias2)

    for i in range(0, 3):
        yt = population[0].test(x[i])
        print(yt)


