
import numpy as np
from Particle import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as cm
import matplotlib
#matplotlib.use("Agg")
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation








swarm = Swarm()
x = np.linspace(psoParam.lBound,psoParam.uBound,30)
y = np.linspace(psoParam.lBound,psoParam.uBound,30)
X1, X2 = np.meshgrid(x,y)


fig = plt.figure()
ax = fig.gca(projection='3d')

z = -(1 + np.cos(12*np.sqrt(np.square(X1)+np.square(X2)))) / -((np.square(X1) + np.square(X2))/2 +1)
surf = ax.plot_surface(X1,X2,z,cmap='viridis',linewidth=0,antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
myanimfig = plt.figure()
CS1 = plt.contourf(X1,X2,z,20)
myanimfig.colorbar(CS1,shrink=0.5,aspect=5)




for particle in swarm: # initialize particles

    particle.velocity = np.random.random(psoParam.numOfVars)
    particle.position = np.random.uniform(psoParam.lBound,psoParam.uBound,psoParam.numOfVars)
    particle.pBest.position = np.random.random(psoParam.numOfVars)*psoParam.vMax
    particle.pBest.o =-np.inf
    particle.plotParticle()
swarm.gBest.position = np.zeros(psoParam.numOfVars) # global best position starts at origin
swarm.gBest.o =-np.inf



for t in range(psoParam.iterations):
    for particle in swarm:
        particle.fitnessFunction()

        if(particle.o > particle.pBest.o):
            particle.pBest.o = particle.o
            particle.pBest.position = particle.position
        if(particle.o > swarm.gBest.o):
            swarm.gBest.o = particle.o
            swarm.gBest.position = particle.position

    w = psoParam.wMax - t*((psoParam.wMax-psoParam.wMin)/psoParam.iterations) # update intertia weight
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
        idx = idx + 1

    idx = 0
    for particle in swarm:
        for x,y in zip(psoParam.moving_x[idx],psoParam.moving_y[idx]):
            particle.moveParticle(x,y)
        idx = idx+1
    plt.pause(0.01)



plt.show()

















