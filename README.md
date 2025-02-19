# Characterisation of Anti-Arrhythmic Drug Effects on Cardiac Electrophysiology using Physics-Informed Neural Networks (ISBI 2024)

Published at the International Symposium on Biomedical Imaging 2024: 
https://doi.org/10.1109/ISBI56570.2024.10635234

## Abstract 
The ability to accurately infer cardiac electrophysiological (EP) properties is key to improving arrhythmia diagnosis and treatment. In this work, we developed a physics-informed neural networks (PINNs) framework to predict how different myocardial EP parameters are modulated by anti-arrhythmic drugs. Using in vitro optical mapping images and the 3-channel Fenton-Karma model, we estimated the changes in ionic channel conductance caused by these drugs. Our framework successfully characterised the action of drugs HMR1556, nifedipine and lidocaine - respectively, blockade of IK, ICa, and INa currents - by estimating that they decreased the respective channel conductance by 31.8±2.7% (p=8.2×10−5), 80.9±21.6% (p=0.02), and 8.6±0.5% (p=0.03), leaving the conductance of other channels unchanged. For carbenoxolone, whose main action is the blockade of intercellular gap junctions, PINNs also successfully predicted no significant changes (p>0.09) in all ionic conductances. Our results are an important step towards the deployment of PINNs for model parameter estimation from experimental data, bringing this framework closer to clinical or laboratory images analysis and for the personalisation of mathematical models.

## Running the code
Run either [`main.py`](main.py) or [`main.ipynb`](main.ipynb) (for running on Google Colab). [`utils.py`](utils.py) and [`generate_plots.py`](generate_plots.py) are supplementary files which should be in the same folder as [`main.py`](main.py)/[`main.ipynb`](main.ipynb).

## Files
#### [`main.py`](main.py)/[`main.ipynb`](main.ipynb) 

The network architecture is defined, compiled and trained. RMSEs are calculated.

#### [`utils.py`](utils.py)

Include various functions for initialising parameter values, loading data, setting the geoemetry, setting the initial guess for parameter estimation, defining the physical equations (1D Fenton-Karma), and setting initial and boundary conditions.

#### [`generate_plots.py`](generate_plots.py)

Include different plotting options. E.g. plotting potential u across time at a particular cell (plot_1D_cell), plotting potential u across cells (spatially) at a particular time (plot_1D_array), and plotting potential across time and space (plot_1D_grid).

#### [`params_plot.py`](params_plot.py)

This is used to visualise the parameter estimation result. It reads the result file and plots parameter estimates across epochs. Not neccesary for model training.
