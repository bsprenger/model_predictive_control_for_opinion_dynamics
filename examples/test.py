import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath('../src'))
from mpc_solvers import SolveStep
from utils import generate_random_dynamics
from fj_models import simulate

np.random.seed(1)
# define problem size
nx = 4
nu = 1

# generate random dynamics
A,B,lambdas,x0 = generate_random_dynamics(nx,nu)

# define parameters
K = 10 # number of time steps

# solve the problem
x,u,cost = SolveStep(A,B,lambdas,x0,K)
print(x[:,1].value)

# plt.plot(np.transpose(x.value))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

ax1.plot(np.transpose(x.value))
ax2.plot(np.transpose(u.value))
# ax2.plot(np.transpose(x.value))
# ax2.set_title('Controlled FJ Model')
# ax3.plot(np.transpose(x2))
# ax3.set_title('Uncontrolled FJ Model')
# ax1.plot(np.transpose(u.value))
# ax1.set_title('Control Node Input')
plt.show()

