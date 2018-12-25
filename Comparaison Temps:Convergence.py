import time
import math
import numpy as np
import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg
import pylab as pl
import matplotlib.pyplot as plt

plt.style.use('ggplot')
#N = 256

def spectral(N):
    '''Algorithme de résolution par méthode spectrale de Fourier-Galerkin. N est la taille du vecteur 1D créé représentant la ligne '''
    compt=time.time()
    # Grille
    N2=int(N/2)
    h = 2*math.pi/N; x = [h*i for i in range(1,N+1)]
    
    # Conditions initiales
    v = [math.sin(y) for y in x]
    alpha = 0.5
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
    for i in range(nplots):
        v_hat = np.fft.fft(v) # passage dans l'espace spectral
        for n in range(plotgap):
            v_hat = v_hat / (1-dt*alpha*k2) # BE (Euler implicite)
        v = np.fft.ifft(v_hat) # retour dans l'espace temporel
        data[i+1,:] = np.real(v) # enregistrement des données
    return(time.time()-compt,data)
    
def implicite(N):
    '''Algorithme de résolution par méthode des éléments finis, en Euler implicite optimisé en matrices creuses'''
    t=time.time()
    h = 2*math.pi/N; x = [h*i for i in range(1,N+1)]
    h = 1/(N+1.0)
    k = h/2
    TFinal = 0.2
    NumOfTimeSteps = int(TFinal/k)
    x = np.linspace(0,1,N+2)
    x = x[1:-1]
    
    # IC
    u = np.transpose(np.mat(np.sin(2*np.pi*x)))
    
    # Opérateur Laplacien
    data = np.ones((3, N))
    data[1] = -2*data[1]
    diags = [-1,0,1]
    L = sparse.spdiags(data,diags,N,N)/(h**2)
    
    # Matrice In
    I = sparse.identity(N)
    
    # Data mat des données
    data = []
    
    for i in range(NumOfTimeSteps):
        # (I - k/2*L) u_new = (I + k/2*L)*u_old
        A = (I -k/2*L)
        b = ( I + k/2*L )*u
        u = np.transpose(np.mat( sparse.linalg.spsolve(A,b)))
        data.append(u)
    return(time.time()-t,np.squeeze(np.asarray(data))) #squeeze et asarray pour renvoyer une mat convenable
    
def comp_tps(N):
    ''' N la N-ième puissance de 2, donne la taille de la grille de discrétisation '''
    tps_spectral,tps_euler=[],[]
    Nliste=[2**k for k in range(0,N)]
    for l in Nliste:
        temp_spectral,temp_euler=[],[]
        for i in range(20):
            temp_spectral+=[spectral(l)[0]]
            temp_euler+=[implicite(l)[0]]
        tps_spectral+=[np.mean(temp_spectral)]
        tps_euler+=[np.mean(temp_euler)]
    plt.clf()
    plt.plot(Nliste,tps_spectral,'g')
    plt.plot(Nliste,tps_euler,'b')
    plt.title('Méthode spectrale en vert, Euler Implicite en bleu')
    plt.grid()
    plt.xlabel("Dimension de la grille (puissance de 2)")
    plt.ylabel("Coût temporel (en secondes)")
    plt.show()
    return(tps_spectral,tps_euler)
    
def comp_cv(N):
    cv_spectral,cv_euler=[],[]
    Nliste=[2**k for k in range(0,N)]
    for l in Nliste:
        fct_test=[0 for i in range(l)]
        temp_spectral,temp_euler=[],[]
        for i in range(20):
            temp_spectral=[np.linalg.norm(spectral(l)[1]-fct_test)]
            temp_euler=[np.linalg.norm(implicite(l)[1]-fct_test)]
        cv_spectral+=[np.mean(temp_spectral)]
        cv_euler+=[np.mean(temp_euler)]
    plt.clf()
    plt.plot(Nliste,cv_spectral,'g')
    plt.plot(Nliste,cv_euler,'b')
    plt.title('Méthode spectrale en vert, Euler Implicite en bleu')
    plt.grid()
    plt.xlabel("Dimension de la grille (puissance de 2)")
    plt.ylabel("Ecart à la solution")
    plt.show()
    return(cv_spectral,cv_euler)
    
def comp_final(N):
    '''compare les deux méthodes au niveau de la cv et du coût temporel'''
    tps_spectral,tps_euler=comp_tps(N)
    cv_spectral,cv_euler=comp_cv(N)
    long_tps,long_cv=len(tps_spectral),len(cv_spectral)
    Nliste1=[2**k for k in range(0,long_tps)]
    Nliste2=[2**k for k in range(0,long_cv)]
    plt.clf()
    plt.subplot(121)
    plt.plot(Nliste1,tps_spectral,'g',label="Spectral")
    plt.plot(Nliste1,tps_euler,'b',label="Euler")
    plt.legend()
    plt.title('Coût temporel')
    plt.grid()
    plt.xlabel("Dimension de la grille (puissance de 2)")
    plt.ylabel("Coût temporel (en secondes)")
    plt.subplot(122)
    plt.plot(Nliste2,cv_spectral,'g',label='Spectral')
    plt.plot(Nliste2,cv_euler,'b',label='Euler')
    plt.legend()
    plt.title('Evolution du résidu selon taille de grille (à temps fixé)')
    plt.grid()
    plt.xlabel("Dimension de la grille (puissance de 2)")
    plt.ylabel("Résidu")
    plt.show()