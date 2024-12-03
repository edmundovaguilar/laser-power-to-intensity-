# Importing libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import seaborn as sns
import scipy
from scipy.optimize import minimize
from scipy.optimize import curve_fit
# ----------------------------
def Real_power(Power):
A = 0.727279836706763
B = -2.0020305219665233
ErA = 8.094833565833314e-05
ErB = 0.0008283305781254659
RP = np.abs(A*Power + B)
error = np.sqrt((ErA**2)*(Power**2)+ErB**2)
return RP,error

# ----------------------------
# translation to Intensity
def get_intensity(Power,erpower,densL):
d_laser = 1850.00
reduction = 6.0
er_laser = 200.0
A_eff = np.pi*((d_laser/reduction)/2.0)**2
errA = (2*(er_laser/d_laser))*A_eff 
Intensity = Power/A_eff
error = np.sqrt((((errA**2)/A_eff**2)+((erpower**2)/Power**2))*Intensity**2)
return Intensity,error

# ----------------------------
def final_intesity(Power,desL):
P, Perr = Real_power(Power)
return get_intensity(P, Perr, desL)

# ----------------------------
# Define the mean and covariance matrix for the 2D Gaussian distribution

from scipy.stats import multivariate_normal
from matplotlib.patches import Rectangle, Circle

# using micrometers
d_laser = 1850.00
Mag = 6.00

mean = np.array([0, 0])
sigma = (d_laser/(Mag*4.0))
cov = np.array([[sigma**2, 0], [0, sigma**2]])

# Create a 2D Laser Gaussian distribution
LGD = multivariate_normal(mean, cov)

# Define the points in the 2D space
x, y = np.mgrid[-300:300:1, -300:300:1]
pos = np.dstack((x, y))

# Compute the PDF at each point
pdf = LGD.pdf(pos)

# -----------------------------

# Calculate the probability from minus infinity to zero along both dimensions
Lx = 720*0.15384
Ly = 540*0.15384

# Define the region of interest
x_min, x_max = -Lx/2, Lx/2
y_min, y_max = -Ly/2, Ly/2

# Integrate the PDF over the defined region
probability = LGD.cdf([x_max, y_max]) - LGD.cdf([x_min, y_max]) - \
LGD.cdf([x_max, y_min]) + LGD.cdf([x_min, y_min])

print("Probability within the region:", probability)