"""
Equation de la chaleur avec une méthode pseudo-spectrale et un Euler implicite:
u_t= \alpha*u_xx
BC = u(0)=0 et u(2*pi)=0 (BC périodique)
IC=sin(x)
"""
    
import time
import math
import numpy as np
import matplotlib.pyplot as plt
compt=time.time()

# Grille
N = 512; h = 2*math.pi/N; x = [h*i for i in range(1,N+1)]
N2=int(N/2)

# Conditions initiales
v = [math.sin(y) for y in x]
alpha = 0.5
t = 0
dt = .001 #Pas temporel

# (ik)^2 Vecteur
I = complex(0,1)
l1 = [I*n for n in range(0,N2)]
l2 = [I*n for n in range(-N2+1,0)]
l = l1+[0]+l2
k = np.array(l)
k2 = k**2;

# Graphique
tmax = 10.0; tplot = 0.1
plotgap = int(round(tplot/dt))
nplots = int(round(tmax/tplot)*N/256)

data = np.zeros((nplots+1,N))
data[0,:] = v
tdata = [t]
for i in range(nplots):
    v_hat = np.fft.fft(v) # passage dans l'espace spectral
    for n in range(plotgap):
        v_hat = v_hat / (1-dt*alpha*k2) # backward Euler timestepping
    v = np.fft.ifft(v_hat) # retour dans l'espace temporel
    data[i+1,:] = np.real(v) # enregistrement des données
    t = t+plotgap*dt # enregistre temps
    tdata.append(t)
print("temps", time.time()-compt)
plt.clf()
plt.imshow(data)
plt.xlabel('x')
plt.ylabel('t (temps)')
plt.colorbar()

## Outils d'analyse spectrale
def analyse_fourier(v):
    '''Donne le signal, la partie réelle et la partie imaginaire de la transformée de fourier'''
    plt.figure()
    plt.subplot(311)
    plt.plot(v)
    v_hat=np.fft.fft(v)
    B = np.append(A, A[0])
    plt.subplot(312)
    plt.plot(np.real(B))
    plt.ylabel("partie réelle")
    plt.subplot(313)
    plt.plot(np.imag(B))
    plt.ylabel("partie imaginaire")
    
def freq_fourier(v):
    '''Donne une représentation  des fréquences de la transformée de Fourier'''
    plt.figure()
    plt.subplot(211)
    plt.plot(v)
    fourier = np.fft.fft(v)
    n = v.size
    freq = np.fft.fftfreq(n, d=dt)
    plt.subplot(212)
    plt.plot(freq, fourier.real, label="real")
    plt.plot(freq, fourier.imag, label="imag")
    plt.legend()
    plt.show()
    
def scale_color(v):
    plt.figure()
    plt.subplot(211)
    plt.plot(v)
    k = np.arange(len(v))
    A = np.fft.fft(v)
    plt.subplot(212)
    x = np.append(k, k[-1]+k[1]-k[0])
    z = np.append(A, A[0])
    X = np.array([x,x])
    y0 = np.zeros(len(x))
    y = np.abs(z)
    Y = np.array([y0,y])
    Z = np.array([z,z])
    C = np.angle(Z)
    plt.plot(x,y,'k')
    plt.pcolormesh(X, Y, C, shading="gouraud", cmap=plt.cm.hsv, vmin=-np.pi, vmax=np.pi)
    plt.colorbar()
    plt.show()
    
def complex_color(v,dt):
    '''Affichage du signal avec la partie complexe en couleurs'''
    a = np.fft.ifftshift(v)
    A = np.fft.fft(a)
    X = dt*np.fft.fftshift(A)
    n = v.size
    freq = np.fft.fftfreq(n, d=dt)
    f = np.fft.fftshift(freq)
    plt.subplot(212)
    x = np.append(f, f[0])
    z = np.append(X, X[0])
    X = np.array([x,x])
    y0 = np.zeros(len(x))
    y = np.abs(z)
    Y = np.array([y0,y])
    Z = np.array([z,z])
    C = np.angle(Z)
    plt.figure()
    plt.plot(x,y,'k')
    plt.pcolormesh(X, Y, C, shading="gouraud", cmap=plt.cm.hsv, vmin=-np.pi, vmax=np.pi)
    plt.colorbar()
    plt.show()
    
## Analyse graphique
def graph_cv(data):
    plt.clf()
    x=[k for k in range(0,len(data[0]))]
    plt.figure(1)
    plt.plot(x,data[0],'g')
    for i in range(1,len(data[:])-2):
        plt.plot(x,data[i],'g')
    plt.grid()
    plt.title('Visualisation de la convergence du signal par application multiple de la méthode spectrale')
    plt.xlabel("Position spatiale sur l'axe x")
    plt.ylabel('Signal')