import numpy as np
import matplotlib.pyplot as plt
datos = np.loadtxt("numeros_20.txt")
training = datos[0:10]
test = datos[10:20]

x = training[:,0]
y = training[:,1]
xtest = test[:,0]
ytest = test[:,1]
xcoef = np.linspace(np.min(x),np.max(x),100)

class PolyFit:
    def __init__(self,degree = 0):
        self.betas = np.ones(degree+1)
        self.degree = degree
    def fit(self,X,Y):
        matriz = []
        for i in range(len(X)):
            agregar = []
            for j in range(self.degree+1):
                agregar.append(X[i]**j)
            matriz.append(agregar)
        inversa = np.linalg.pinv(matriz)
        self.betas = np.dot(inversa,Y)
    def predict(self,X):
        ceros = np.zeros(len(X))
        for l in range(len(self.betas)):
            ceros += self.betas[l]*X**l
        return ceros
    def score(self,X,Y):
        ypred = self.predict(X)
        resultado = 0
        for i in range(len(X)):
            resultado += (ypred[i]-Y[i])**2
        return(np.sqrt(2*resultado/10))
    
contador = 1
plt.figure(figsize = (10,10))
for i in [0,1,3,9]:
    poli = PolyFit(i)
    poli.fit(x,y)
    ycoef = poli.predict(xcoef)
    plt.subplot(2,2,contador)
    plt.plot(xcoef,ycoef, color = "r")
    plt.scatter(training[:,0],training[:,1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("M={}".format(ordenes[i]))
    contador += 1
plt.subplots_adjust(hspace=0.5)
plt.savefig("ajustes.png")

ErmsTest = []
ErmsTraining = []
for i in range(10):
    poli = PolyFit(i)
    poli.fit(x,y)
    ErmsTraining.append(np.log10(poli.score(x,y)))
    ErmsTest.append(np.log10(poli.score(xtest,ytest)))
plt.figure()
plt.plot(ordenes, ErmsTraining, label = "Training", color = "b", marker = "o", markerfacecolor ="none",markeredgecolor="blue",ms=10)
plt.plot(ordenes, ErmsTest, label = "Test",color = "r",marker = "o", markerfacecolor ="none",markeredgecolor="red",ms=10)
plt.legend()
plt.ylabel("logErms")
plt.xlabel("M")
plt.savefig("Erms.png")