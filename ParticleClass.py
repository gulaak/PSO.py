import numpy as np
import matplotlib.pyplot as plt
class pBest:
    def __init__(self):
        self.o = None
        self.position = np.array([None, None])

    def __str__(self):
        return f"O: {self.o} position: {self.position}"


class Particle(object):

    def __init__(self, x=0, y=0):
        self.position = np.array([x, y])
        self.velocity = np.array([None, None])
        self.o = None
        self.pBest = pBest()

    def __str__(self):
        return f"Position: {self.position} V: {self.velocity} O: {self.o} pBest: {self.pBest}"

    def fitnessFunction(self):
        x = self.position[0]
        y = self.position[1]
        # temp = self.position + 5
        # dim = len(self.position)
        # self.o = np.sum(np.square(temp) - 10*np.cos(2*np.pi * temp)) + 10*dim
        self.o = -(1 + np.cos(12*np.sqrt(np.square(x)+np.square(y)))) / ((np.square(x) + np.square(y))/2 +1)

    def plotParticle(self):
        self.handle, = plt.plot(self.position[0], self.position[1], marker='$*$', markersize=11, color='k')
        return self.handle,

    def setPos(self, x, y):
        self.position = np.array([x, y])

    def moveParticle(self,x=None,y=None):
        if(x==None or y==None):
            self.handle.set_ydata(self.position[1])
            self.handle.set_xdata(self.position[0])
        else:
            self.handle.set_ydata(y)
            self.handle.set_xdata(x)



    def deleteParticle(self):
        self.handle.remove()


class Swarm:
    def __init__(self, x=0, y=0):
        self.swarm = [Particle() for i in range(psoParam.numOfParticles)]  # initialize all particles
        self.gBest = pBest()

    def __str__(self):
        mystring = [i.__str__() for i in self.swarm]
        return str(mystring)

    def __len__(self):
        return len(self.swarm)

    def getParticle(self, idx):
        return self.swarm[idx]

    def __iter__(self):
        for i in self.swarm:
            yield (i)

    def __len__(self):
        return len(self.swarm)


class psoParam:
    numOfParticles = 2
    iterations = 200
    movingLength = 20
    c1 = 2
    c2 = 2
    w=0.7298
    wMax = 0.9
    wMin = 0.2
    numOfVars = 2
    vMax = 0.6
    uBound = 2
    lBound = -2
    gBest = list()
    moving_x = np.zeros(movingLength*numOfParticles).reshape(numOfParticles,movingLength)
    moving_y = np.zeros(movingLength*numOfParticles).reshape(numOfParticles,movingLength)
    first_loc = np.zeros(numOfParticles*numOfVars).reshape(numOfParticles,numOfVars)
    second_loc = np.zeros(numOfParticles*numOfVars).reshape(numOfParticles,numOfVars)


    @classmethod
    def setParticles(cls, num):
        cls.numOfParticles = num

    def setIterations(cls, iterations):
        cls.iterations = iterations
