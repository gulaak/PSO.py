import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as cm
from matplotlib import animation
from drawnow import *
def ObjFunction(X):
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

    def ObjFunction(self):
        temp = self.position + 5
        dim = len(self.position)
        self.o = np.sum(np.square(temp) - 10*np.cos(2*np.pi * temp)) + 10*dim

    def plotParticle(self):
        self.handle, = plt.plot(self.position[0],self.position[1], marker='.',markersize=11,color='k')
    def moveParticle(self,x,y):
        self.handle.set_ydata(y)
        self.handle.set_xdata(x)







class Swarm:
    def __init__(self,x=0,y=0):
       self.swarm = [Particle() for i in range(psoParam.numOfParticles)] #initialize all particles
       self.gBest = pBest()
    def __str__(self):
        mystring = [i.__str__() for i in self.swarm]
        return str(mystring)

    def __len__(self):
        return len(self.swarm)

    def setPos(self,idx,x,y):
        self.swarm[idx].position = np.array([x,y])

    def getParticle(self,idx):
        return self.swarm[idx]


    def __iter__(self):
        for i in self.swarm:
            yield(i)


class psoParam:
    numOfParticles = 36
    iterations = 200
    movingLength = 20
    c1 =2
    c2 = 2
    wMax = 0.9
    wMin = 0.2
    vMax = 6
    moving_x = np.zeros(movingLength*numOfParticles).reshape(numOfParticles,movingLength)
    moving_y = np.zeros(movingLength*numOfParticles).reshape(numOfParticles,movingLength)
    first_loc = np.zeros(numOfParticles*2).reshape(numOfParticles,2)
    second_loc = np.zeros(numOfParticles*2).reshape(numOfParticles,2)
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
# x1 = np.linspace(0,2.0*np.pi,n1)
# x2 = np.linspace(0,2.0*np.pi,n2)
X1, X2 = np.meshgrid(x,y)
# z = np.sin(X1)*np.cos(X2)
z = np.zeros(900).reshape(30,30)
for k1 in range(len(X1)):
    for k2 in range(len(X1)):
        X = np.array([X1[k1,k2], X2[k1,k2]])
        z[k1,k2] = ObjFunction(X)
breaks = np.linspace(-1,1,11)

plt.figure()
CS1 = plt.contour(X1,X2,z,20)

#plt.colorbar(ticks= breaks,orientation='vertical')

x = np.linspace(-10,10,6)
y = x
idx = 0
for t1 in range(len(x)):
    for t2 in range(len(y)):
        swarm.setPos(idx,x[t1],y[t2])
        idx = idx+1



for particle in swarm: # initialize particles
   particle.velocity = np.random.random(2)
   #particle.position = (np.random.random(2) * 2*10) -10
   particle.pBest.position = np.random.random(2)*psoParam.vMax
   particle.pBest.o = np.inf
   particle.plotParticle()



swarm.gBest.position = np.zeros(2) # global best position starts at origin
swarm.gBest.o = np.inf
for t in range(psoParam.iterations):
    for particle in swarm:
        particle.ObjFunction()

        if(particle.o < particle.pBest.o):
            particle.pBest.o = particle.o
            particle.pBest.position = particle.position
        if(particle.o < swarm.gBest.o):
            swarm.gBest.o = particle.o
            swarm.gBest.position = particle.position

    w = psoParam.wMax - t*((psoParam.wMax-psoParam.wMin)/psoParam.iterations) # update intertia weight
    idx=0
    for particle in swarm:
        #first_vel = particle.velocity
        particle.velocity = w * particle.velocity + psoParam.c1 * np.random.random(2) * \
        (particle.pBest.position - particle.position) + \
        psoParam.c2 * np.random.random(2) *(swarm.gBest.position - particle.position)

       # second_vel = particle.velocity

        idxx = np.where(particle.velocity > psoParam.vMax)
        particle.velocity[idxx] = psoParam.vMax * np.random.random()

        idxx = np.where(particle.velocity < -psoParam.vMax)
        particle.velocity[idxx] = -psoParam.vMax * np.random.random()

        psoParam.first_loc[idx,:] = particle.position
        particle.position = particle.position + particle.velocity
        psoParam.second_loc[idx,:] = particle.position

        idxx = np.where(particle.position > ub)
        particle.position[idxx] = ub

        idxx = np.where(particle.position <lb)
        particle.position[idxx] = lb


        psoParam.moving_x[idx,:] = np.linspace(psoParam.first_loc[idx,0],psoParam.second_loc[idx,0],psoParam.movingLength)
      #  print(psoParam.moving_x)
        psoParam.moving_y[idx,:] = np.linspace(psoParam.first_loc[idx,1],psoParam.second_loc[idx,1],psoParam.movingLength)
        #moving_vx = np.linspace(first_vel[0],second_vel[0],psoParam.movingLength)
        #moving_vy = np.linspace(first_vel[1],second_vel[1],psoParam.movingLength)
        idx = idx+1
    plt.pause(0.05)
    for inc in range(psoParam.movingLength):

        for i in range(psoParam.numOfParticles):

            swarm.getParticle(i).moveParticle(psoParam.moving_x[i,inc],psoParam.moving_y[i,inc])











