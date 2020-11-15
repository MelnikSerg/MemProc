import math
import matplotlib.pyplot as plt
from memory_process import Process


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

#input() #uncomment to prevent the console window disappearing when the program terminates