# MemProc
Generating non-markovian random processes with the prescribed memory function
Theoretical description of the method: https://arxiv.org/abs/1904.03514

Functions:
Gen_mu_Delta          - Genetare process X(t) by the memory function mu(t) = delta(t-T)
Gen_mu_Step           - Genetare process X(t) by the memory function mu(t<T) = mu0
Gen_mu                - Genetare process X(t) by the given memory function mu(t<T) (modify the string "mu = ..." for the function needed)
SaveDataToFile_CSV    - Save Data to the Comma-Separated Valued (.csv) file
SaveDataToFile_Column - Save Data to the text file as one column
CalcCor               - Calculate the correlation function of the X(t) process

-----
To run the program you may need to install Python first: https://www.python.org/downloads/
