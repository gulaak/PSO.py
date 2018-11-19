import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation

def fitnessFunction(X):
    X = X+5
    dim = len(X)
    return np.sum(np.square(X)- 10*np.cos(2*np.pi*X)) + 10*dim
	

class pBest:
    def __init__(self):
        self.o = None
        self.position = np.array([None, None])
    def __str__(self):
        return f"O: {self.o} position: {self.position}"

class Particle(object):

    def __init__(self,x =0,y = 0):
        self.position = np.array([x , y])
        self.velocity = np.array([None ,None])
        self.o = None
        self.pBest = pBest()


    def __str__(self):
        return f"Position: {self.position} V: {self.velocity} O: {self.o} pBest: {self.pBest}"
	
	
    def fitnessFunction(self):
        temp = self.position + 5
        dim = len(self.position)
        self.o = np.sum(np.square(temp) - 10*np.cos(2*np.pi * temp)) + 10*dim

    def plotParticle(self):
        self.handle, = plt.plot(self.position[0],self.position[1], marker='$*$',markersize=11,color='k')
        return self.handle,
		
    def setPos(self,x,y):
        self.position=np.array([x,y])
    
    def moveParticle(self):
        self.handle.set_ydata(self.position[1])
        self.handle.set_xdata(self.position[0])
    def deleteParticle(self):
        self.handle.remove()
   
		
	

		
	





class Swarm:
    def __init__(self,x=0,y=0):
       self.swarm = [Particle() for i in range(psoParam.numOfParticles)] #initialize all particles
       self.gBest = pBest()
	   
    def __str__(self):
        mystring = [i.__str__() for i in self.swarm]
        return str(mystring)

    def __len__(self):
        return len(self.swarm)
    

    def getParticle(self,idx):
        return self.swarm[idx]


    def __iter__(self):
        for i in self.swarm:
            yield(i)
    def __len__(self):
        return len(self.swarm)
  


class psoParam:
    numOfParticles = 36
    iterations = 100
    movingLength = 20
    c1 =2
    c2 = 2
    wMax = 0.9
    wMin = 0.2
    numOfVars = 2
    vMax = 6
    gBest = list()
    @classmethod
    def setParticles(cls,num):
        cls.numOfParticles = num
    def setIterations(cls,iterations):
        cls.iterations = iterations





n1 = 1010
n2 = 51
ub=10
lb = -10

swarm = Swarm()
x = np.linspace(-10,10,30)
y = np.linspace(-10,10,30)
X1, X2 = np.meshgrid(x,y)
z = np.zeros(900).reshape(30,30)
for k1 in range(len(X1)):
    for k2 in range(len(X1)):
        X = np.array([X1[k1,k2], X2[k1,k2]])
        z[k1,k2] = fitnessFunction(X)
breaks = np.linspace(-1,1,11)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X1,X2,z,cmap="cool",linewidth=0,antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)


myanimfig = plt.figure()
CS1 = plt.contour(X1,X2,z,20)

def init():
    artistarray = []
    x = np.linspace(-10,10,int(np.sqrt(len(swarm))))
    y = x
    idx = 0
    for t1 in range(len(x)):
        for t2 in range(len(y)):
            swarm.getParticle(idx).setPos(x[t1],y[t2])  # place particles in uniform distribution of search space.
            idx +=1


    for particle in swarm: # initialize particles

       particle.velocity = np.random.random(psoParam.numOfVars)
       #particle.position = (np.random.random(2) * 2*10) -10
       particle.pBest.position = np.random.random(psoParam.numOfVars)*psoParam.vMax
       particle.pBest.o =0 #np.inf
       particle.plotParticle()
    swarm.gBest.position = np.zeros(psoParam.numOfVars) # global best position starts at origin
    swarm.gBest.o =0 #np.inf
    


def update(iteration):
   
        for particle in swarm:
            particle.fitnessFunction()

            if(particle.o > particle.pBest.o):
                particle.pBest.o = particle.o
                particle.pBest.position = particle.position
            if(particle.o > swarm.gBest.o):
                swarm.gBest.o = particle.o
                swarm.gBest.position = particle.position

        w = psoParam.wMax - iteration*((psoParam.wMax-psoParam.wMin)/psoParam.iterations) # update intertia weight
        idx=0
        for particle in swarm:

            particle.velocity = w * particle.velocity + psoParam.c1 * np.random.random(psoParam.numOfVars) * \
            (particle.pBest.position - particle.position) + \
            psoParam.c2 * np.random.random(psoParam.numOfVars) *(swarm.gBest.position - particle.position)

            idxx = np.where(particle.velocity > psoParam.vMax)
            particle.velocity[idxx] = psoParam.vMax * np.random.random()

            idxx = np.where(particle.velocity < -psoParam.vMax)
            particle.velocity[idxx] = -psoParam.vMax * np.random.random()

            #psoParam.first_loc[idx,:] = particle.position
            particle.position = particle.position + particle.velocity

            #psoParam.second_loc[idx,:] = particle.position

            idxx = np.where(particle.position > ub)
            particle.position[idxx] = ub

            idxx = np.where(particle.position <lb)
            particle.position[idxx] = lb
        
            particle.moveParticle()
            #psoParam.moving_x[idx,:] = np.linspace(psoParam.first_loc[idx,0],psoParam.second_loc[idx,0],psoParam.movingLength)
            #psoParam.moving_y[idx,:] = np.linspace(psoParam.first_loc[idx,1],psoParam.second_loc[idx,1],psoParam.movingLength)

            idx = idx+1
        if(iteration==psoParam.iterations):
            for particle in swarm:
                particle.deleteParticle()
        
                 

      

numOfIterations = psoParam.iterations
anim = FuncAnimation(myanimfig, update, np.linspace(1,numOfIterations,numOfIterations),init_func=init,interval=100)
plt.show()















