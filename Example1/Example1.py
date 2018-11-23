import random
from Particle import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import animation




def fitnessFunction(X):
    X = X+5
    dim = len(X)
    return np.sum(np.square(X)- 10*np.cos(2*np.pi*X)) + 10*dim


swarm = Swarm()
x = np.linspace(-10,10,30)
y = np.linspace(-10,10,30)
X1, X2 = np.meshgrid(x,y)
z = np.zeros(30*30).reshape(30,30)
for k1 in range(len(X1)):
     for k2 in range(len(X1)):
        X = np.array([X1[k1,k2], X2[k1,k2]])
        z[k1,k2] = fitnessFunction(X)


fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X1,X2,z,cmap="viridis",linewidth=0,antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)


fig = plt.figure()
CS1 = plt.contourf(X1,X2,z,20)

print(len(swarm))
x = np.linspace(-10,10,int(np.sqrt(len(swarm))))
y = x
idx = 0

for t1 in range(len(x)):
    for t2 in range(len(y)):
        swarm.getParticle(idx).setPos(x[t1],y[t2])  # place particles in uniform distribution of search space.
        idx +=1
 


for particle in swarm: # initialize particles

   particle.velocity = np.random.random(psoParam.numOfVars)
   particle.pBest.position = np.random.random(psoParam.numOfVars)*psoParam.vMax
   particle.pBest.o = np.inf
   particle.plotParticle()



swarm.gBest.position = np.zeros(psoParam.numOfVars) # global best position starts at origin
swarm.gBest.o = np.inf
for t in range(psoParam.generations):
    for particle in swarm:
        particle.fitnessFunction()

        if(particle.o < particle.pBest.o):
            particle.pBest.o = particle.o
            particle.pBest.position = particle.position
        if(particle.o < swarm.gBest.o):
            swarm.gBest.o = particle.o
            swarm.gBest.position = particle.position

    w = psoParam.wMax - t*((psoParam.wMax-psoParam.wMin)/psoParam.generations) # update intertia weight
    idx=0
    for particle in swarm:

        particle.velocity = w * particle.velocity + psoParam.c1 * np.random.random(2) * \
        (particle.pBest.position - particle.position) + \
        psoParam.c2 * np.random.random(psoParam.numOfVars) *(swarm.gBest.position - particle.position)

        idxx = np.where(particle.velocity > psoParam.vMax)
        particle.velocity[idxx] = psoParam.vMax * np.random.random()

        idxx = np.where(particle.velocity < -psoParam.vMax)
        particle.velocity[idxx] = -psoParam.vMax * np.random.random()

        psoParam.first_loc[idx,:] = particle.position
        particle.position = particle.position + particle.velocity

        psoParam.second_loc[idx,:] = particle.position

        idxx = np.where(particle.position > psoParam.ub)
        particle.position[idxx] = psoParam.ub

        idxx = np.where(particle.position <psoParam.lb)
        particle.position[idxx] = psoParam.lb

        psoParam.moving_x[idx,:] = np.linspace(psoParam.first_loc[idx,0],psoParam.second_loc[idx,0],psoParam.movingLength)
        psoParam.moving_y[idx,:] = np.linspace(psoParam.first_loc[idx,1],psoParam.second_loc[idx,1],psoParam.movingLength)

        idx = idx+1

    index = 0
    for particle in swarm:
            for x,y in zip(psoParam.moving_x[index],psoParam.moving_y[index]):

                particle.moveParticle(x,y)
            index = index +1
    plt.pause(0.001)
    psoParam.gBest.append(swarm.gBest.o) # create an array of global bests for graph


plt.figure()
plt.plot(np.linspace(1,psoParam.generations,psoParam.generations),psoParam.gBest)
plt.title('Global Best Fitness vs Generations')

plt.show()












