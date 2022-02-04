# Operator-Learning

This repository studies the cost-accuracy trade-off in operator learning with neural networks



There are four applications, Navier-Stokes equation (`Navier-Stokes`), Helmholtz equation (`Helmholtz-high`), structural equation (`Solid`), and advection equation (`advection-dc-low`).

The neural network related functions are in the `nn` folder. 

Each application has one folder, which includes the PDE solver and different NN training folders including `PCA`,  `DeepONet`, `PARA` and `FNO`.

    Generate data : generate_data_script, NN-Data-Par.jl
    
    Error plot : Data-NN-Plot.jl
    
    Postprocess worst and median predition data: Map-Plot_script
    
    Sample and plot input output data, and plot the worst and median preditions: Map-Plot.jl
