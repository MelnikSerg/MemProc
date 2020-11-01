import math
import random
import csv
import numpy
import matplotlib.pyplot as plt
from enum import Enum

class Process:

    class MemoryType(Enum):
        NULL = 0
        DELTA = 1
        STEP = 2
        EXP = 3
        FUNCTION = 4
        LIST_MEM = 5
        LIST_COR = 6

    def __init__(self, n_points = 0, memory_type = MemoryType.NULL, T = 1.0, mu0 = 0.0, nu = 0.0, sigma = 1.0, dt = 1.0, vals_mem = []):
        self.memory_type = memory_type
        self.T = T
        self.mu0 = mu0
        self.nu = nu
        self.sigma = sigma
        self.dt = dt
        self.n_points = n_points
        self.vals_mem = vals_mem[:]
        self.x = []
    
    def generate(self):
        """Generate process x(t)"""

        random.seed(version=2)  #initializing random generator
        self.x = []             #clear the list
        self.x.append(0)        #start value is zero
        
        c = self.mu0*self.dt
        e = math.exp(-self.dt/self.T)
        Tint = int(self.T/self.dt + 0.5)
        sig_dt = self.sigma*math.sqrt(self.dt)

        #main loop
        mem = 0.0
        for i in range(1, self.n_points):
            #recalc memory part
            if self.memory_type == Process.MemoryType.EXP:
                mem = (mem + self.x[-1]*c)*e
            elif self.memory_type == Process.MemoryType.DELTA:
                if(i >= Tint):
                    mem = self.x[i-Tint]*self.mu0
            elif self.memory_type == Process.MemoryType.STEP:
                mem += self.x[i-1]*self.mu0*self.dt
                if(i > Tint): mem -= self.x[i-1-Tint]*self.mu0*self.dt
            elif self.memory_type == Process.MemoryType.FUNCTION:
                mem = 0.0
                for r in range(1, min(i, Tint)):
                    t = r*self.dt
                    mu = math.exp(-1.2*t)   #MEMORY FUNCTION/mu0
                    mem += self.x[i-r]*mu
                mem *= self.mu0*self.dt
            elif self.memory_type == Process.MemoryType.LIST_MEM:
                mem = 0
                for r in range(1, min(i, len(self.vals_mem))):
                    mem += self.x[i-r]*self.vals_mem[r]
                mem *= self.dt

            x0 = self.x[-1] - (self.nu*self.x[-1] + mem)*self.dt    #new center value
            self.x.append(random.gauss(x0, sig_dt))                 #generate with Gauss distribution
    
    def set_mu0_to_diffusiveborder(self):
        if self.memory_type == Process.MemoryType.EXP:
            self.mu0 = -self.nu/self.dt*(math.exp(self.dt/self.T) - 1)
        elif self.memory_type == Process.MemoryType.DELTA:
            self.mu0 = -self.nu
        elif self.memory_type == Process.MemoryType.STEP:
            Tint = int(self.T/self.dt + 0.5)
            self.mu0 = -self.nu/(Tint*self.dt)
        else:
            print('The method set_mu0_to_diffusiveborder is not supported for this memory type')

    def save_process(self, fileName):
        """Save process to the text file as two column (t; x)"""

        auto_round_pow = int(-math.log10(self.dt))+1

        with open(fileName, 'w') as f:
            t = 0
            for x in self.x:
                f.write(str(round(t,auto_round_pow)) + '\t' + str(x) + '\n')
                t += self.dt
    
    def autocorrelation(self, t_max):
        """Correlator Calculation"""
        cor = []
        n_t = int(t_max/self.dt)
        x_aver2 = numpy.average(self.x) ** 2
        for i_t in range(n_t+1):
            Ct = 0
            for i in range(len(self.x)-i_t):
                Ct += self.x[i]*self.x[i+i_t]
            cor.append(Ct/(len(self.x)-i_t) - x_aver2)
        return cor
    
    def gen_by_corr(self, C):
        """Genetare process X(t) by the given correlation function C(t) (defined as list)"""
        #make a matrix for the Linear equations system
        N = len(C)-1
        A = [0.0]*N
        B = [0.0]*N
        for n in range(N):
            A[n]=[0.0]*N
            for r in range(N):
                A[n][r]=C[abs(n-r)]
            B[n] = C[n]-C[n+1]
    
        #solving the system for dt^2 * mu[t]
        t2mu = numpy.linalg.solve(A, B)

        #sqrt(tau)* sigma
        tsigma = math.sqrt(2.0*(C[0]-C[1]))

        #main loop
        self.x = [0]
        for n in range(1, self.n_points):
            #calc Xc - gaussian center for the next X value
            Xc = self.x[-1]
            for r in range(min(N,n)):
                Xc -= t2mu[r]*self.x[-1-r]
            #generate new value by gaussian
            self.x.append(random.gauss(Xc, tsigma))

    def сalc_сorr_norm(self, t_max):
        """Calculate normalized C[t] (C[0]=1) corresponding to given mu[t] and nu"""

        n_t = int(t_max/self.dt + 0.5)

        #put nu into mu[0]
        self.vals_mem[0] = self.nu/self.dt
    
        #calculate h(t)
        h = [1.0]
        for i in range(n_t):
            dh_t = 0.0
            for j in range(min(i,len(self.vals_mem))):
                dh_t -= self.vals_mem[j]*h[-1-j]
            dh_t *= self.dt
            h.append(h[-1]+dh_t*self.dt)

        #calculate correlator
        corr = []
        for i in range(n_t):
            Ct = 0
            for j in range(len(h)-i):
                Ct += h[j]*h[j+i]
            Ct *= self.dt
            corr.append(Ct)

        #normalize
        kC = 1.0/corr[0]
        corr = numpy.array(corr) * kC
        
        return corr
        
    def save_data_to_file_csv(file_name, data):
        """Save data[] list to the csv file"""
        with open(file_name, 'w') as w_file:
            writer = csv.writer(w_file)
            writer.writerow(data)
        
    def save_data_to_file_column(file_name, data):
        """Save data[] list to the text file as one column"""
        with open(file_name, 'w') as w_file:
            for x in data:
                w_file.write(str(x) + '\n')

    def save_data_to_file_twocolumn(file_name, data, t0, dt):
        """Save data[] list to the text file as two column (t; x)"""        
        t = t0
        with open(file_name, 'w') as w_file:
            for x in data:
                w_file.write(str(t) + '\t' + str(x) + '\n')
                t += dt

    def calc_snr(self):
        """Signal-to-Noise ratio"""

        # fit frequency w
        xx = 0
        xx2 = 0
        for i in range(1, len(self.x) - 1):
            xx += self.x[i]*self.x[i]
            xx2 += self.x[i]*(self.x[i+1] - self.x[i]*2 + self.x[i-1])
        if (xx == 0): return 0
        if (xx2 > 0): return 0
        w = math.sqrt(-xx2/(xx*self.dt**2))
        T = 2*math.pi/w
        
        # calc SNR
        prev_ampl = 0
        last_ampl = 0
        i_last_zero = 0
        ampl2 = 0
        d_ampl2 = 0
        for i in range(1, len(self.x)):
            last_ampl = max(last_ampl, abs(self.x[i]))
            if (numpy.sign(self.x[i]) != numpy.sign(self.x[i-1])) and (i - i_last_zero > T*0.25):
                ampl2 += last_ampl ** 2
                d_ampl2 += (last_ampl - prev_ampl) ** 2
                prev_ampl = last_ampl
                last_ampl = 0
                i_last_zero = i

        if d_ampl2 == 0:
            return 0
        else:
            return ampl2/d_ampl2

    def variance_by_ansamble(self, t_max, n_ansambles):
        n_points_prev = self.n_points
        self.n_points = int(t_max/self.dt)+1
        variance = [0]*self.n_points
        for iter in range(n_ansambles):
            self.generate()
            for i in range(self.n_points):
                variance[i] += pow(self.x[i], 2)
        for i in range(self.n_points):
            variance[i] /= n_ansambles

        self.n_points = n_points_prev
        return variance

    def draw_process(self):
        t = numpy.arange(0, len(self.x)*self.dt, self.dt)
        plt.plot(t, self.x)
        plt.xlabel('t')
        plt.ylabel('X')

###======================= EXAMPLES ===========================================

## ------------- generate the process with given parameters (direct problem) --

#brownian motion
proc_brownian = Process(memory_type = Process.MemoryType.NULL, dt = 0.1, n_points = 1000)
proc_brownian.generate()
proc_brownian.draw_process()

#Ornstein–Uhlenbeck process
proc_ornuhl = Process(memory_type = Process.MemoryType.NULL, nu = 1, dt = 0.1, n_points = 1000)
proc_ornuhl.generate()
proc_ornuhl.draw_process()

#process with stepwise non-local memory
proc_mem = Process(memory_type=Process.MemoryType.STEP, T=1, mu0=4.5, nu=0, sigma=1, dt=0.1, n_points=1000)
proc_mem.generate()
proc_mem.draw_process()
#save the process to file
proc_mem.save_process('process.dat')

## ------------- generate the process with given autocorrelation (inverse problem) --

# fill the correlator list by a desired function
T_CORR = 10.0   #maximum time of correlations
N_CORR = 100    #number of correlator values
dt = T_CORR/N_CORR
C = []
for nt in range(N_CORR+1):
    t = dt*nt
    C.append(10*(1.0-t)*math.exp(-1.1*t))

# Generating the process
proc = Process(n_points = 1000, dt = dt)
proc.gen_by_corr(C)
proc.draw_process()
plt.show()

#input()