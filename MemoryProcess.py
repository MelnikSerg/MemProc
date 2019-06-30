import math
import random
import csv

X = []  #list of X values

def Gen_mu_Delta(T, mu0, nu, sigma, dt, Npoints):
    """Genetare process X(t) by the memory function mu(t) = delta(t-T)"""
    
    random.seed(version=2)  #initializing random generator
    X.clear                 #clear the list
    X.append(0)             #start value is zero
    Tint = int(T/dt)        #T in units of dt

    #main loop
    for i in range(1, Npoints):
        X0 = X[i-1]*(1.0-nu*dt)
        if(i > Tint): X0 -= X[i-Tint]*mu0*dt
        X.append(random.gauss(X0, sigma))

def Gen_mu_Step(T, mu0, nu, sigma, dt, Npoints):
    """Genetare process X(t) by the memory function mu(t<T) = mu0"""
    
    random.seed(version=2)  #initializing random generator
    X.clear                 #clear the list
    X.append(0)             #start value is zero
    Tint = int(T/dt)        #T in units of dt

    #main loop
    mem = 0.0   #memory component
    for i in range(1, Npoints):
        mem += X[i-1]*mu0*dt
        if(i > Tint): mem -= X[i-1-Tint]*mu0*dt
        X0 = X[i-1] - (nu*X[i-1] + mem)*dt
        
        X.append(random.gauss(X0, sigma))

def Gen_mu(T, nu, sigma, dt, Npoints):
    """Genetare process X(t) by the given memory function mu(t<T)"""
    
    random.seed(version=2)  #initializing random generator
    X.clear                 #clear the list
    X.append(0)             #start value is zero
    Tint = int(T/dt)        #T in units of dt

    #main loop
    for i in range(1, Npoints):
        mem = 0.0   #memory component
        for r in range(1, min(i, Tint)):
            t = r*dt
            mu = 0.1*math.exp(-1.2*t)
            mem += X[i-r]*mu*dt
        
        X0 = X[i-1] - (nu*X[i-1] + mem)*dt
        
        X.append(random.gauss(X0, sigma))
        print(X[-1])

def SaveXToFile_CSV(FileName):
    """Save X(t) list to the csv file"""
    with open(FileName, 'w') as WFile:
        writer = csv.writer(WFile)
        writer.writerow(X)
        
def SaveXToFile_Column(FileName):
    """Save X(t) list to the text file as one column"""
    f = open(FileName, 'w')
    for x in X:
        f.write(str(x) + '\r\n')
    f.close()

#Gen_mu_Delta(1, 1.0, 0.0, 1, 0.1, 10000)
#Gen_mu_Step(1, 4, 0, 1, 0.1, 1000)
Gen_mu(1, 1, 1, 0.1, 1000)
print(X[-1])

#SaveXToFileCSV('delta.csv')
SaveXToFile_Column('X(t).dat')