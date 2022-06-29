from multiprocessing.connection import wait
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

import time
from IPython.display import clear_output

#DEFINIR PROBLEMA (CIRCULOS QUE HAY QUE SEPARAR)
n = 500     #Cantidad de datos              //Filas
p = 2       #Caracteristicas de cada dato   //Columnas

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.07)

Y = Y[:,np.newaxis]

'''plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], color="blue")
plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], color="red")
plt.axis('equal')
plt.show()'''

#CAPA DE NEURONAS
class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f                                          #Funcion de activacion
        self.b = np.random.rand(1, n_neur) * 2 - 1                  #Definir bias para cada neurona
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1             #Definir W para cada conexion de cada neurona

#FUNCIONES DE ACTIVACION
sigm = (lambda x: 1 / (1 + np.e ** (-x)),                           #Funcion sigmoide
        lambda x: x * (1 - x))                                      #Su derivada

relu = lambda x: np.maximum(0,x)                                    #Funcion relu     

#RED
def create_nn(topology, act_f):
    nn = []

    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l+1], act_f))

    return nn

topology = [p, 4, 8, 1]

#FUNCION ENTRENAR
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),                  #Error cuadratico medio
           lambda Yp, Yr: (Yp - Yr))                                #Su derivada                

def train(neural_net, X, Y, l2_cost, lr=0.05, train = True):
    out = [(None, X)]
    
    #forward pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)

        out.append((z, a))

    if train:
        deltas = []

        for l in reversed(range(0, len(neural_net))):
            #Backward pass

            z = out[l+1][0]
            a = out[l+1][1]

            if l == len(neural_net) - 1:
                #Delta de la ultima capa
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                #Delta de todas las otras capas
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W

            #Gradient descent
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr
    
    return out[-1][1]

neural_n = create_nn(topology, sigm)

loss = []

for i in range(2500):
    pY = train(neural_n, X, Y, l2_cost)

    if i % 200 == 0:
        loss.append(l2_cost[0](pY, Y))

        res = 50

        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)

        _Y = np.zeros((res, res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]
        
        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")

        plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], color="blue")
        plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], color="red")

        clear_output(wait=True)
        plt.show()
        time.sleep(0.5)




    


