# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:48:49 2019

@author: aria
"""

import matplotlib.pyplot as plt
import numpy as np
import constants as c

class ensemble: 
    def __init__(self, numberOfParticles, isElectron, isFirstOrder, isPositiveVel): 
        self.numberOfParticles = numberOfParticles
        self.isElectron = isElectron
        self.isFirstOrder = isFirstOrder 
        self.isPositiveVel = isPositiveVel
        self.particles = self.createParticles(numberOfParticles, isElectron, isPositiveVel)            
        self.positions = [particle.position for particle in self.particles]
        self.velocities = [particle.velocity for particle in self.particles]

    def createParticles(self, numberOfParticles, isElectron, isPositiveVel):
        initialPositions = np.linspace(c.LOW, c.HIGH, numberOfParticles)
        particles = []
        for position in initialPositions: 
            pert = 0.001 * position * (self.isPositiveVel * 2.0 - 1.0) * self.isElectron / (2.0*np.pi) #sinusoidal and out of phase by 2pi for electrons
            position += pert
            position = self.applyBC(position)
            particles.append(self.particle(position, numberOfParticles, isElectron, isPositiveVel))
        return particles           

    def particlePush(self, isElectron):
        charge = (1.0 - 2.0 * isElectron) #just gives sign of the charged particle
        for particle in self.particles:
            particle.velocity += c.DELTA_T * charge * particle.Efield 
            particle.position += c.DELTA_T * particle.velocity 
            particle.position = self.applyBC(particle.position) 
            particle.velocityHistory.append(particle.velocity) 
            particle.trajectory.append(particle.position)
        self.positions = [particle.position for particle in self.particles]
        self.velocities = [particle.velocity for particle in self.particles]

    def applyBC(self, position): 
        while np.abs(position) >= c.HIGH: 
            if position >= c.HIGH : 
                position = c.LOW + (position + c.LOW) + c.ETA
            elif position <= c.LOW: 
                position = c.HIGH - np.abs(position + c.HIGH) - c.ETA
        return position
    
#    def plotPhaseSpaceInTime(self, totalSteps):
#        for timeStep in range(totalSteps + 2):          
#            positionAtTimestep = [particle.trajectory[timeStep] for particle in self.particles]   
#            velocityAtTimestep = [particle.velocityHistory[timeStep] for particle in self.particles]   
#            plt.scatter(positionAtTimestep, velocityAtTimestep)
#        plt.xlabel("Position")
#        plt.ylabel("Velocity")
#        plt.title("Phase Space for " + str(self.numberOfParticles) + ' Test Particles Initialized Uniformly $x\in (-\pi , \pi)$')#part3
#        filename = "prob3phaseSpaceN" + str(self.numberOfParticles)  
#        plt.savefig(filename) 
#        plt.show()
#        plt.close()
      
    def plotEnergies(self, totalSteps):
        totalE = []
        totalKE = []
        totalEE = []
        for timeStep in range(totalSteps+2): #+2 because initialization process
            currentE = 0.0            
            currentKE = 0.0
            currentEE = 0.0            
            for particle in self.particles:
                currentEE += 0.5 * particle.EfieldHistory[timeStep]**2
                currentKE += 0.5 * particle.velocityHistory[timeStep]**2
                currentE += 0.5 * particle.EfieldHistory[timeStep]**2 + 0.5 * particle.velocityHistory[timeStep]**2
            totalE.append(currentE)
            totalEE.append(currentEE)
            totalKE.append(currentKE)
        plt.plot(range(totalSteps+2), totalE, label="Total Energy")        
        plt.plot(range(totalSteps+2), totalKE, label="Kinetic Energy")
        plt.plot(range(totalSteps+2), totalEE, label="Electric Field Energy")
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Energies')
        plt.title('Energies Diagnostic for ' + str(self.numberOfParticles) + ' Test Particles Initialized Uniformly $x\in (-\pi , \pi)$')        
        filename = "prob3EnergiesFor" + str(self.numberOfParticles) +"test" 
        plt.savefig(filename)
        plt.show()
        plt.close()

    class particle: 
        def __init__(self, position, numberOfParticles, isElectron, isPositiveVel):
            self.position = position
            self.charge = ((-2) * isElectron + 1) / numberOfParticles 
            self.velocity = self.initializeVelocity(isPositiveVel, isElectron)
            self.trajectory = [self.position]
            self.velocityHistory = [self.velocity]
            self.Efield = 0.0
            self.EfieldHistory = [self.Efield]            
            
        def initializeVelocity(self, isPositiveVel, isElectron):
            vel = (2.0 * isPositiveVel - 1.0) * c.VPRIME if isElectron else 0.0
            return vel
                           
class myGrid:
#    def __init__(self, background, test, numberOfPoints, isFirstOrder): #rewrite to have arb number of ensembles
    def __init__(self, ensembles, numberOfPoints, isFirstOrder):
        self.numberOfPoints = numberOfPoints
        self.deltaX = (c.HIGH - c.LOW) / numberOfPoints
        self.isFirstOrder = isFirstOrder
        self.Ainv = self.calculateAInverse(numberOfPoints, self.deltaX)
        self.gridPoints = self.createPoints(numberOfPoints, self.deltaX, isFirstOrder)
        self.rho = np.asarray([gridPoint.rho for gridPoint in self.gridPoints])   
        self.positions = np.asarray([gridPoint.position for gridPoint in self.gridPoints])     
        self.phi = -np.matmul(self.Ainv, self.rho) 
        self.Efields = self.calculateEField(self.phi, self.deltaX)

    def calculateAInverse(self, numberOfPoints, deltaX):
        A = np.zeros((numberOfPoints, numberOfPoints))
        A[0, (numberOfPoints-1)] = 1.0/(deltaX**2) #wtf these bc's
        for j in range(numberOfPoints):
            A[(numberOfPoints-1),j] = 1.0/(deltaX**2)
        for k in range(numberOfPoints):
            A[k,k] = -2.0/(deltaX**2)
            i = k % numberOfPoints
            if (k < numberOfPoints -1 and (k+1)%numberOfPoints!=0) or k == 0:
                A[k,k+1] =  1.0/(deltaX**2)
            if (k != 0 and i != 0) or k == 1:
                A[k,k-1] =  1.0/(deltaX**2)
        A[(numberOfPoints-1),(numberOfPoints-1)] = 0
        return np.linalg.inv(A)
    
    def createPoints(self, numberOfPoints, deltaX, isFirstOrder):
        positions = np.linspace(c.LOW, c.HIGH, numberOfPoints)
        points = []
        for position in positions: 
            points.append(myGrid.gridPoint(position, deltaX, isFirstOrder))
        return points
    
    def calculateEField(self, phi, deltaX):
        E = np.ones_like(phi) #make it the same size        
        for j in range(np.size(phi)): #initialize 
            jhi = j+1 if j+1 < np.size(phi) else 0
            jlo = j-1 if j-1 >= 0 else -1
            E[j] = -(phi[jhi] - phi[jlo])/(2*deltaX)
        return E

    #step 1 
    def particleWeighting(self):
        for gridPoint in self.gridPoints:
            gridPoint.rho = gridPoint.calculateTotalRho(gridPoint.position, ensembles, deltaX, isFirstOrder)
        self.rho = np.asarray([gridPoint.rho for gridPoint in self.gridPoints])

    #step 2
    def fieldSolve(self):
        self.phi = -np.matmul(self.Ainv, self.rho) 
        self.Efields = self.calculateEField(self.phi, self.deltaX)

    #step 3 
    def fieldWeighting(self, ensembles, deltaX, isFirstOrder):   
        if (isFirstOrder):        
            for ensemble in ensembles:
                for particle in ensemble.particles:
                    for j in range(np.size(myGrid.Efields)): 
                        jhi = j+1 if j+1 < np.size(myGrid.Efields) else 0
                        if (particle.position < myGrid.positions[jhi] and particle.position >= myGrid.positions[j]):
                            particle.Efield = (myGrid.positions[jhi] - particle.position)*myGrid.Efields[j]/deltaX 
                            particle.Efield += (particle.position - myGrid.positions[j]) * myGrid.Efields[jhi]/deltaX                    
                            particle.EfieldHistory.append(particle.Efield)
                            break
                        elif (particle.position == myGrid.positions[jhi]):
                            particle.Efield = myGrid.Efields[jhi]   
                            particle.EfieldHistory.append(particle.Efield)                    
                            break
        elif (not isFirstOrder):
            for ensemble in ensembles:
                for particle in ensemble.particles:
                    for j in range(np.size(myGrid.Efields)-1): 
                        jhi = j+1 if j+1 < np.size(myGrid.Efields) else 0
                        if (abs(particle.position - myGrid.positions[jhi] ) <= abs(particle.position - myGrid.positions[j]) ) or (particle.position == myGrid.positions[jhi]):
                            particle.Efield = myGrid.Efields[j+1] 
                            particle.EfieldHistory.append(particle.Efield)
                            break
                        else:
                            particle.Efield = myGrid.Efields[j] 
                            particle.EfieldHistory.append(particle.Efield)
                            break
        
    def sanityCheck(self, rho, phi, Efields, isFirstOrder):
        plt.plot(self.positions, rho, label="rho")        
        plt.plot(self.positions, phi, label="phi")
        plt.plot(self.positions, Efields, label="E")
        plt.legend()
        plt.xlabel('Position')
        plt.ylabel('Magnitudes')
        plt.title('Diagnostic for ' + str(isFirstOrder*1) + 'th/st Order Weightings')
        plt.close()
        
    class gridPoint:
        def __init__(self, position, deltaX, isFirstOrder):
            self.position = position
            self.isFirstOrder = isFirstOrder
            self.rho = self.calculateTotalRho(position, ensembles, deltaX, isFirstOrder)
            
        def calculateTotalRho(self, position, ensembles, deltaX, isFirstOrder):
            rho = 0
            for ensemble in ensembles:
                for particle in ensemble.particles:                    
                    rho += self.calculateRhoFromCharge(position, particle.position, particle.charge, deltaX, isFirstOrder)
            return rho
    
        def calculateRhoFromCharge(self, gp, pp, charge, deltaX, isFirstOrder):           
            if (gp == c.HIGH and pp < c.LOW + deltaX):
                dist = abs(c.LOW - pp)
                b = abs(deltaX - dist) 
                return charge * b / (deltaX ** 2)
            dist = abs(gp-pp)
            if (dist > deltaX):
                return 0.0
            elif (isFirstOrder):
                b = abs(deltaX - dist) 
                return charge * b / (deltaX ** 2)
            elif (not isFirstOrder):
                return (1.0 * charge if (dist < deltaX / 2.0) else 0.0)               
            
def evolveSystem(ensembles, Grid, deltaX, isFirstOrder, steps):
    for i in range(steps):
        Grid.particleWeighting()
        Grid.fieldSolve()
        Grid.fieldWeighting(ensembles, deltaX, isFirstOrder)
        for ensemble in ensembles:        
            if ensemble.isElectron:        
                ensemble.particlePush(True)        

def plotEnergies(totalSteps, ensembles):
    totalE = []
    totalKE = []
    totalEE = []
    for timeStep in range(totalSteps+2): #+2 because initialization process
        currentE = 0.0            
        currentKE = 0.0
        currentEE = 0.0      
        for ensemble in ensembles:
            for particle in ensemble.particles:
                vel = particle.velocityHistory[timeStep] if ensemble.isElectron else 0.0
                E = particle.EfieldHistory[timeStep] #this guy is out of range -- probably for the background
                currentEE += 0.5 * E**2 #particle.EfieldHistory[timeStep]**2 
                currentKE += 0.5 * vel**2 
                currentE += 0.5 * E**2 + 0.5 * vel**2  
        totalE.append(currentE)
        totalEE.append(currentEE)
        totalKE.append(currentKE)
    plt.plot(range(totalSteps+2), totalE, label="Total Energy")        
    plt.plot(range(totalSteps+2), totalKE, label="Kinetic Energy")
    plt.plot(range(totalSteps+2), totalEE, label="Electric Field Energy")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Energies')
    plt.title('Energies Diagnostic for Two Stream Instability') 
    filename = "EnergiesForTwoStreamInstab"
    plt.savefig(filename)
#    plt.show()
    plt.close()

def plotPhaseSpaceOfEnsembles(timeSteps, ensembles):
    for timeStep in range(1, int(timeSteps/2)):    
        for ensemble in ensembles:
            if ensemble.isElectron:
                positionAtTimestep = [particle.trajectory[timeStep] for particle in ensemble.particles]   
                velocityAtTimestep = [particle.velocityHistory[timeStep] for particle in ensemble.particles]   
                plt.scatter(positionAtTimestep, velocityAtTimestep)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Phase Space for Two Streaming Electrons")#part3
    filename = "prob3phaseSpaceForESpecies" 
    plt.savefig(filename) 
    #plt.show()
    plt.close()
    for timeStep in range(int(timeSteps/2), timeSteps):    
        for ensemble in ensembles:
            if ensemble.isElectron:
                positionAtTimestep = [particle.trajectory[timeStep] for particle in ensemble.particles]   
                velocityAtTimestep = [particle.velocityHistory[timeStep] for particle in ensemble.particles]   
                plt.scatter(positionAtTimestep, velocityAtTimestep)
    #plt.show()
    plt.close()

#constants           
numberOfGridPoints = 512
isFirstOrder = True#False
deltaX = (c.HIGH - c.LOW)/numberOfGridPoints
numberOfElectrons = 512
backgroundParticleNumber = 2 * numberOfElectrons #background ion population is twice the size of the electron population
totalSteps = 420

#ensmebles to work with 
background = ensemble(backgroundParticleNumber, False, isFirstOrder, False) #treat background as ions, test as electrons 
posElectrons = ensemble(numberOfElectrons, True, isFirstOrder, True)
negElectrons = ensemble(numberOfElectrons, True, isFirstOrder, False)
ensembles = []
ensembles.append(background)
ensembles.append(posElectrons)
ensembles.append(negElectrons)

#grid definition
myGrid = myGrid(ensembles, numberOfGridPoints, isFirstOrder) #making the grid does particle weight and field solving

#test particles
myGrid.fieldWeighting(ensembles, deltaX, isFirstOrder)
for ensemble in ensembles:
    if ensemble.isElectron == True:
        ensemble.particlePush(True)

evolveSystem(ensembles, myGrid, deltaX, isFirstOrder, totalSteps)
plotEnergies(totalSteps, ensembles)
plotPhaseSpaceOfEnsembles((totalSteps +2), ensembles)