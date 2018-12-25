import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import time
 
t=time.time()
# Number of internal points
N = 512
 
# Calculate Spatial Step-Size
h = 1/(N+1.0)
 
# Create Temporal Step-Size, TFinal, Number of Time-Steps
k = h/2
TFinal = 0.2
NumOfTimeSteps = int(TFinal/k)
 
# Create grid-points on x axis
x = np.linspace(0,1,N+2)
x = x[1:-1]
 
# Initial Conditions
u = np.transpose(np.mat(10*np.sin(2*np.pi*x)))
 
# Second-Derivative Matrix
data = np.ones((3, N))
data[1] = -2*data[1]
diags = [-1,0,1]
L = sparse.spdiags(data,diags,N,N)/(h**2)
 
# Identity Matrix
I = sparse.identity(N)
 
# Data for each time-step
data = []
 
for i in range(NumOfTimeSteps):
    # Solve the System: (I - k/2*L) u_new = (I + k/2*L)*u_old
    A = (I -k/2*L)
    b = ( I + k/2*L )*u
    u = np.transpose(np.mat( sparse.linalg.spsolve(A,b)))
    data.append(u)
    
print(time.time()-t)
plt.clf()
plt.imshow(np.squeeze(np.asarray(data))) #transforme le format matrix en format array pour imshow
plt.xlabel('x')
plt.ylabel('t (temps)')
plt.colorbar()

#http://jkwiens.com/2010/01/02/finite-difference-heat-equation-using-numpy/