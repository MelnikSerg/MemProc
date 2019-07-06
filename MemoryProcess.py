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
        #print(X[-1])

def SaveDataToFile_CSV(FileName, Data):
    """Save Data[] list to the csv file"""
    with open(FileName, 'w') as WFile:
        writer = csv.writer(WFile)
        writer.writerow(Data)
        
def SaveDataToFile_Column(FileName, Data):
    """Save Data[] list to the text file as one column"""
    f = open(FileName, 'w')
    for x in Data:
        f.write(str(x) + '\r\n')
    f.close()

def CalcCor(t_max):
    """Correlator Calculation"""
    C = []
    for t in range(1, t_max):
        Ct = 0
        for i in range(len(X)-t):
            Ct += X[i]*X[i+t]
        C.append(Ct/(len(X)-t))
        print(t, ":", C[-1])
    return C



#Gen_mu_Delta(1, 1.5, 0.0, 1, 0.1, 10000)
#Gen_mu_Step(1, 1, 0, 1, 0.1, 1000)
Gen_mu(1, 5, 1, 0.1, 100000)
print("Last value X[", len(X), "] = ", X[-1])
#SaveDataToFile_Column('X(t).dat', X)

C = CalcCor(100)
SaveDataToFile_Column('C(t).dat', C)
