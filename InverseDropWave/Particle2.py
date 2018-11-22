import random
import numpy as np

from ParticleClass import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as cm
import matplotlib
#matplotlib.use("Agg")
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers





def fitnessFunction(X1,X2):
   # X = X+5
    #dim = len(X)
    #return np.sum(np.square(X)- 10*np.cos(2*np.pi*X)) + 10*dim
   return np.cos(X1) + np.sin(X2)
	









swarm = Swarm()
x = np.linspace(psoParam.lBound,psoParam.uBound,30)
y = np.linspace(psoParam.lBound,psoParam.uBound,30)
X1, X2 = np.meshgrid(x,y)
#z = np.zeros(900).reshape(30,30)
#for k1 in range(len(X1)):
 #   for k2 in range(len(X1)):
  #      X = np.array([X1[k1,k2], X2[k1,k2]])
   #     z[k1,k2] = np.square(X) #fitnessFunction(X)
#breaks = np.linspace(-1,1,11)

fig = plt.figure()
ax = fig.gca(projection='3d')

z = -(1 + np.cos(12*np.sqrt(np.square(X1)+np.square(X2)))) / -((np.square(X1) + np.square(X2))/2 +1)
surf = ax.plot_surface(X1,X2,z,cmap="summer",linewidth=0,antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)


myanimfig = plt.figure()
CS1 = plt.contour(X1,X2,z,20)
myanimfig.colorbar(CS1,shrink=0.5,aspect=5)

def init(): # initialization for animation 

    #x = np.linspace(psoParam.lBound,psoParam.uBound,int(np.sqrt(len(swarm))))
   # y = x
    #idx = 0
    #for t1 in range(len(x)):
       # for t2 in range(len(y)):
           # swarm.getParticle(idx).setPos(x[t1],y[t2])  # place particles in uniform distribution of search space.
            #idx +=1


    for particle in swarm: # initialize particles

       particle.velocity = np.random.random(psoParam.numOfVars)
       particle.position = np.random.uniform(psoParam.lBound,psoParam.uBound,psoParam.numOfVars)
       particle.pBest.position = np.random.random(psoParam.numOfVars)*psoParam.vMax
       particle.pBest.o =-np.inf
       particle.plotParticle()
    swarm.gBest.position = np.zeros(psoParam.numOfVars) # global best position starts at origin
    swarm.gBest.o =-np.inf
    


def update(iteration):  # updates each particle in the swarm specified by some interval
   
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

            psoParam.first_loc[idx,:] = particle.position
            particle.position = particle.position + particle.velocity
            psoParam.second_loc[idx,:] = particle.position

            idxx = np.where(particle.position > psoParam.uBound)
            particle.position[idxx] = psoParam.uBound

            idxx = np.where(particle.position <psoParam.lBound)
            particle.position[idxx] = psoParam.lBound
        
            #particle.moveParticle()
            psoParam.moving_x[idx,:] = np.linspace(psoParam.first_loc[idx,0],psoParam.second_loc[idx,0],psoParam.movingLength)
            psoParam.moving_y[idx,:] = np.linspace(psoParam.first_loc[idx,1],psoParam.second_loc[idx,1],psoParam.movingLength)

            idx = idx+1
        idx = 0
        for particle in swarm:

            for x,y in zip(psoParam.moving_x[idx],psoParam.moving_y[idx]):
                particle.moveParticle(x,y)
            idx = idx+1


        if(iteration==psoParam.iterations):
            for particle in swarm:
                particle.deleteParticle()
        
                 

      

numOfIterations = psoParam.iterations
anim = FuncAnimation(myanimfig, update, np.linspace(1,numOfIterations,numOfIterations),init_func=init,interval=50)
#plt.show()

anim.save('anim.gif',writer='imagemagick',fps=30)















