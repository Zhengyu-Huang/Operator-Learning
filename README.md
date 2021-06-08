# Operator-Learning

This repository compares two different ways to 
learn parametric operator


There are three applications, Helmholtz equation, Navier-Stokes equation, and Burger's equation.

The NN structure and data processing functions are in the `nn` folder. 

Each application has one folder, which includes the PDE solver and NN training scripts (in the `nn` subfolder). Each `nn` folder has data generation file, and `direct` and `PCA` folders for two different learning approaches.

The data is generated and saved as `.npy` matrix files:
$$\theta:  Nd\times N_\theta$$
$$NN(\theta):  Nx \times Ny \times Nd$$
and *pytorch* is used for the training.