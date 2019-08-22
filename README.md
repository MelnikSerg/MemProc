# MemProc

Generating non-markovian random processes with the prescribed memory or correlation function
- The method is based on the Mori-Zwanzig and autoregressive models. Its theoretical description: https://arxiv.org/abs/1904.03514
- To run the program you may need to install **Python** first: https://www.python.org/downloads/

## Functions:
- Gen_mu_Delta          - Genetare process X(t) by the memory function mu(t) = μ0 𝛿(t-T)
- Gen_mu_Step           - Genetare process X(t) by the memory function mu(t<T) = μ0
- Gen_mu                - Genetare process X(t) by the given memory function mu(t<T) (modify the string "mu = ..." for the function needed)
- Gen_Cor               - Generate process X(t) with the given correlator C(t)
- SaveDataToFile_CSV    - Save Data to the Comma-Separated Valued (.csv) file
- SaveDataToFile_Column - Save Data to the text file as one column
- CalcCor               - Calculate the correlation function of the X(t) process
