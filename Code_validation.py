import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
uns = np.ones(250)
zeros = np.zeros(1000-250)
x = np.hstack((uns,zeros))
harmonique = 20





def carre1():
    T = 1/1000
    dt = T / 1000  # Increased sampling rate
    t1 = T *10  
    t = np.arange(0, t1, dt)
    u = np.zeros((20,10000),dtype=complex)

    # création du signal avec scipy.signal
    # f_frequency = 1/T  # Frequency corresponding to the period
    duty_cycle = 0.5  # 50% duty cycle for the entire 4-second period
    x2 =  5* signal.square(2 * np.pi / T*t, duty=0.5) +5

    # affichage du signal

  
 
    w0 = 2*np.pi/T
    
    #calcul des exponentielle
    for k in range(0,harmonique):
        u[k,:] = np.exp(-1j*(k+1)*w0*t)
# plt.plot(t, np.real(u[0,:]))
#    plt.plot(t, np.real(u[1,:]))
    X = np.zeros(21,dtype=complex)
 #calcul des coefficients
    X[0] = np.sum(x2)/18000
    for k in range(0,20):
        X[k+1] = np.sum(x2*u[k,:])/18000
    
    y = np.abs(X[0])*np.sum(x2*u[k,:])/18000

    plt.stem(range(0,21), np.abs(X))

    y = np.abs(X[0]) * np.ones_like(t)/18000
    for k in range(1,21):
        y += 2*np.abs(X[k]) * np.cos(k*w0*t + np.angle(X[k]))
    plt.figure()
    plt.plot(t,y)
    plt.plot(t, x2, label='Repeating Square Wave')
    plt.xlim(-0.05,0.05)
    
    plt.show()
#carre1()


def carre2():
    dt = 1 / 1000  # Increased sampling rate
    t = np.arange(-20 * np.pi, 20* np.pi, dt)
    u = np.zeros((20,125664),dtype=complex)
    # création du signal avec scipy.signal
    # f_frequency = 1/T  # Frequency corresponding to the period
    duty_cycle = 0.2  # 50% duty cycle for the entire 4-second period
    x2 = 0.5*signal.square((0.2*t + (np.pi/5)), duty=duty_cycle)+0.5
   

    w0 = 2*np.pi/10*np.pi
    
    #calcul des exponentielle
    for k in range(0,harmonique):
        u[k,:] = np.exp(-1j*(k+1)*w0*t)
# plt.plot(t, np.real(u[0,:]))x
#    plt.plot(t, np.real(u[1,:]))
    X = np.zeros(21,dtype=complex)
 #calcul des coefficients
    X[0] = np.sum(x2)*dt
    for k in range(0,20):
        X[k+1] = np.sum(x2*u[k,:])*dt
    
    y = np.abs(X[0])*np.sum(x2*u[k,:])*dt

    #plt.stem(range(0,21), np.abs(X))

    y = np.abs(X[0]) * np.ones_like(t)*dt
    for k in range(1,21):
        y += 2*np.abs(X[k]) * np.cos(k*w0*t + np.angle(X[k]))
    plt.figure()
    plt.plot(t,y)
    plt.plot(t, x2, label='Repeating Square Wave')
    plt.xlim(-2*np.pi,2*np.pi)    
    plt.show()
#carre2()


def triangle(): 
    T = 2*np.pi
    dt = 1 / 1000  # Increased sampling rate
    t = np.arange(-2 * 10*np.pi, 2 * 10*np.pi, 1/1000)
    u = np.zeros((20,125664),dtype=complex)
    # création du signal avec scipy.signal
    # f_frequency = 1/T  # Frequency corresponding to the period
   # 50% duty cycle for the entire 4-second period
    x2 = 0.5 * signal.sawtooth(1 * t ) +0.5

    # affichage du signal
    debut = -6
    fin = 6
    ndata = (debut-fin)*1000
    # plt.plot(t, x2, label='Repeating Square Wave')
    plt.ylim(-1.1, 1.1)
    plt.xlim(-4*np.pi, 4*np.pi)
   

    w0 = 1
    
    #calcul des exponentielle
    for k in range(0,5):
        u[k,:] = np.exp(-1j*(k+1)*w0*t)
# plt.plot(t, np.real(u[0,:]))
#    plt.plot(t, np.real(u[1,:]))
    X = np.zeros(21,dtype=complex)
 #calcul des coefficients
    X[0] = np.sum(x2)/12000
    for k in range(0,20):
        X[k+1] = np.sum(x2*u[k,:])*dt
    
    y = np.abs(X[0])*np.sum(x2*u[k,:])*dt

    plt.stem(range(0,harmonique + 1), np.abs(X))

    y = np.abs(X[0]) * np.ones_like(t)
    for k in range(1,harmonique + 1):
        y += 2*np.abs(X[k]) * np.cos(k*w0*t + np.angle(X[k]))
    plt.figure()
    plt.plot(t,y)
    plt.xlim(-2*np.pi,2*np.pi)
    plt.show()
    

#triangle()


# f0 = np.arange(0,2, 0.01) 
# Q = [1,3,5,7]
# k = np.pi
# K = np.abs(k)
# H = -K/(1+1j*Q[0]*(f0-1/f0))


# # Adjusted starting point to avoid division by zero
# f = 1500
# #gain de H(F) vs fn
# plt.figure(figsize=(8, 6))
# for index in np.arange(0, len(Q), 1):
#     H = -K / (1 + 1j * Q[index] * (f0 - 1 / f0))
#     Ha = np.abs(H)
#     plt.semilogx(f0, Ha, label='Q' + str(Q[index]))


# # for i in range(0,5):
# #     H = -i/(1+1j*Q[0]*(f0-1/f0))
# #     Ha = np.abs(H)
# #     print(H)
# plt.grid()
# plt.xlabel('f0')
# plt.ylabel('gain')

# plt.legend()
# plt.show()



# #calcul du gain en db
# plt.figure(figsize=(8,6))
# for index in np.arange(0,len(Q),1):
#     H = -K/ (1+1j*Q[index]*(f0 -1 / f0)) 
#     Ha = np.abs(H)
#     gain_dB = 20*np.log10(np.abs(H))
#     plt.semilogx(f0,gain_dB, label='Q'+str(Q[index]))
#     K2 = np.abs(K)*(1/np.sqrt(1+Q[index]*(f0 -(1/f0) )))
#     print (K2)
# plt.grid(True)
# plt.xlabel('f0')
# plt.ylabel('Gain (dB)')
# plt.legend()
# plt.ylim([-80,15])
# plt.show()


  
def Convolution():
    
    f_x = 15   #frequence sinus
    f_xa = 30   #frequence sinus redresse => x(t)
    nbPeriode = 20
    temps20Periode = nbPeriode/f_xa #longeuur axe de temps

    dt = temps20Periode/1000    #increment infinitimal
    t = np.arange(0,temps20Periode,dt)  #echelle du temps

    x = np.sin(2*np.pi*f_x*t)   #signal d'entree
    xa = np.abs(x)  #(sinus redresse avec f=30hz)
    h = 30*np.exp(-30*t) #reponse impulsionnelle dun

    y = np.convolve(xa,h) * dt #convolution de x(t) et de y(t)

    t_extended = np.arange(0, len(y) * dt, dt)[:len(y)]
    
    plt.figure()
    plt.plot(t, xa, label='Sinus redresse de 30hz x(t)')
    plt.plot(t_extended[:len(x)], y[:len(x)], label='Pseudo-DC y(t)')
    plt.xlim([0,temps20Periode])
    plt.ylim([0,1])
    plt.grid(True)
    plt.show()
    
   

#Convolution()
    
def Bode_passeBas1():

    fB = 100
    f = np.logspace(0, 6)
    mag = (1/np.sqrt(1+(f/fB)**2))
    mag_15Hz = (1/np.sqrt(1+(15/fB)**2))
    mag_10kHz = (1/np.sqrt(1+(10000/fB)**2))
    dB_15Hz=  20*np.log10(mag_15Hz)
    dB_10kHz=  20*np.log10(mag_10kHz)
    dB = 20*np.log10(mag)
    phase = -np.arctan(f/fB)

    print('Gain à 15Hz en db: ',dB_15Hz)
    print('Gain à 10kHz en db',dB_10kHz)
    
    # plt.subplot(3, 1, 1)
    # plt.semilogx()
    # plt.ylim(0, 3.5)
    # plt.ylabel("Amplitude (V)")
    # plt.vlines(fB, 0, 1, colors="g")
    # plt.plot(f,mag)

    plt.subplot(2, 1, 1)
    plt.semilogx()
    plt.title("Phase du passe-bas")
    plt.ylabel("Phase (rad)")
    plt.vlines(fB, -np.pi/2, 0, colors="g")
    plt.plot(f,phase)
    
    plt.subplot(2, 1, 2)
    plt.semilogx()
    plt.ylim(-45, 1)
    plt.ylabel("Amplitude (dB)")
    plt.vlines(fB, -45, 0, colors="g")
    plt.hlines(fB, 1, 0, colors="g")
    plt.legend
    plt.title("Gain passe-bas")
    plt.plot(f,dB)

  

    plt.show()


Bode_passeBas1()
def Bode_passeHaut():


    fB = 6.39
    f = np.logspace(0, 6)
    mag = (f / fB) / np.sqrt(1 + (f / fB)**2)
    mag_15Hz = (15 / fB)  / np.sqrt(1 + (15 / fB)**2)
    mag_epsilon = (2 / fB)/ np.sqrt(1 + (2 / fB)**2)
    dB_15Hz = 20 * np.log10(mag_15Hz)
    dB_epsilon = 20 * np.log10(mag_epsilon)
    dB = 20 * np.log10(mag)
    phase = -np.arctan(f / fB)

    print('Gain à 15Hz en db: ', dB_15Hz)
    print('Gain à epsilon en db', dB_epsilon)

    plt.subplot(2, 1, 1)
    plt.semilogx(f, phase)
    plt.ylabel("Phase (rad)")
    plt.vlines(fB, -np.pi/2, 0, colors="g")

    plt.subplot(2, 1, 2)
    plt.semilogx(f, dB)
    plt.ylim(-25, 1)
    plt.ylabel("Amplitude (dB)")
    plt.vlines(fB, -24, 0, colors="g")
    plt.title("Résultat du passe-haut")
    #plt.hlines(dB_15Hz, 1, 0, colors="g")

    plt.show()



Bode_passeHaut()

def Convolution_passeBas():
    f_x = 15   #frequence sinus
    f_xa = 30   #frequence sinus redresse => x(t)
    nbPeriode = 20
    temps20Periode = nbPeriode/f_xa #longeuur axe de temps

    dt = temps20Periode/1000    #increment infinitimal
    t = np.arange(0,temps20Periode,dt)  #echelle du temps

    x = np.sin(2*np.pi*f_x*t)   #signal d'entree
    xa = np.abs(x)  #(sinus redresse avec f=30hz)
    h = 30*np.exp(-30*t) #reponse impulsionnelle dun

    y = np.convolve(xa,h) * dt #convolution de x(t) et de y(t)

    t_extended = np.arange(0, len(y) * dt, dt)[:len(y)]
    
    plt.figure()
    plt.plot(t, xa, label='Sinus redressé de 30Hz x(t)')
    plt.plot(t_extended[:len(x)], y[:len(x)], label='Pseudo-DC y(t)')
    plt.xlim([0,temps20Periode])
    plt.ylim([0,1])
    plt.grid(True) 
    plt.legend()
    plt.title("Résultat de la convolution du 2e passe-bas")
    plt.show()
   

    
    # ax3 = plt.subplot(3,1,3)
    # ax3.plot(2*np.pi*30*tt, y)
    # ax3 = plt.subplot(3,1,3)
    # ax3.plot(2*np.pi*30*tt, y)
   

Convolution_passeBas()


def calcul_labo():
    T = 1/1000 
    t1 = T *1
    dt = T/1000
    t = np.arange(0, t1, dt)
    x1 = 10* signal.square(2 * np.pi / T*t, duty=0.5)
    N1 = len(x1) # nombre d'échantillon de x1
    dt1 = 1/1000 # pas d'échantillonnage
    t1 = 10*T # pour que la longueur(x1)=longueur(t1)
    # On va afficher pour vérifier
    plt.figure(figsize=(8, 6))
    plt.plot(t, x1)
    plt.axis([0,0.05, -2, 1])
    Xk_pos = np.zeros(21, dtype=complex) # initialisation du vecteur Xk_pos
    nk_pos = np.zeros(21)
    T1 = dt1*N1 # calcul de la période du signal x1
    f1 = 1/T1 # calcul de la fréquence de x1
    w01 = 2*np.pi*f1
    X0 = (1/T1)*np.sum(x1)*dt1
    for k in np.arange(1, 21, 1):
        Xk = (1/T1)*np.sum(x1*np.exp(-1j*k*w01*t1))*dt1
        Xk_pos[k-1] = Xk # on rempli le tableau des Xk positifs
        nk_pos[k-1] = k # on rempli le tableau des nk positifs
    
    Xk_neg = np.conj(Xk_pos)
    Xk_neg = Xk_neg[::-1] # on inverse l'ordre des éléments du tableau
    nk_neg = -1*nk_pos[::-1]
    f_fond = (1/N1)*(1/dt1) # pour vérifier la fréquence à la fondamentale (k=1)
    print(f_fond) # affiche la valeur de la fréquence fondamentale
    nk = np.concatenate((nk_neg, [0], nk_pos))
    X_k = np.concatenate((Xk_neg, [X0], Xk_pos))
    plt.figure()
    plt.stem(nk, np.abs(X_k))
    plt.axis([-20, 20, 0, 0.7])
    plt.figure()
    plt.stem(nk, np.angle(X_k))
    plt.axis([-20, 20, -4, 4])
    plt.show()
#calcul_labo()

def convolution_bas1():
      
    f_x = 15   #frequence sinus
    f_xa = 30   #frequence sinus redresse => x(t)
    nbPeriode = 20
    temps20Periode = nbPeriode/f_xa #longeuur axe de temps

    dt = temps20Periode/1000    #increment infinitimal
    t = np.arange(0,temps20Periode,dt)  #echelle du temps

    x = np.sin(2*np.pi*f_x*t)   #signal d'entree
    xa = np.abs(x)  #(sinus redresse avec f=30hz)
    h = 628*np.exp(-628*t) #reponse impulsionnelle dun

    y = np.convolve(xa,h) * dt #convolution de x(t) et de y(t)

    t_extended = np.arange(0, len(y) * dt, dt)[:len(y)]
    
    plt.figure()
    plt.plot(t, xa, label='Sinus redresse de 30hz x(t)')
    plt.plot(t_extended[:len(x)], y[:len(x)], label='Pseudo-DC y(t)')
    plt.xlim([0,temps20Periode])
    plt.ylim([0,1])
    plt.grid(True)
    plt.show()
    
    # ax3 = plt.subplot(3,1,3)
    # ax3.plot(2*np.pi*30*tt, y)
    # ax3 = plt.subplot(3,1,3)
    # ax3.plot(2*np.pi*30*tt, y)
#convolution_bas1()

def passeBas1():
    f0 = np.arange(0,100, 0.01) 
    Q = [1,3,5,7]
    k = 0.7082
    K = np.abs(k)
    H = -K/(1+1j*Q[0]*(f0-1/f0))
    h_0 = np.abs(-K/(1+1j*Q[0]*(1-1/1)))
    fB= 5000
    f = np.logspace(10**-2, 10**2, 1000)
    x2 = 1/np.sqrt(1+(f/fB)**2)
    print('valeur de H_0 =' , h_0)

    # Adjusted starting point to avoid division by zero
    f = 1500
    #gain de H(F) vs fn
    plt.figure(figsize=(8, 6))
    # for index in np.arange(0, len(Q), 1):
    #     H = -K / (1 + 1j * Q[index] * (f0 - 1 / f0))
    #     Ha = np.abs(H)
    #     plt.semilogx(f0, Ha, label='Q' + str(Q[index]))


    # for i in range(0,5):
    #     H = -i/(1+1j*Q[0]*(f0-1/f0))
    #     Ha = np.abs(H)
    #     print(H)
    plt.grid()
    plt.xlabel('f0')
    plt.ylabel('gain')
    plt.ylim([-0.1,4])
    plt.legend()

    plt.show()

    for k in range(0,15):
        x2 = 1/np.sqrt(1+(f/fB)**2)


    H_15 = np.abs(-K/ (1+1j*Q[index]*(15 -1 / 15)))

    Amplitude_entree = 3
    Amplitude_sortie = H_15
    H_15_gain = Amplitude_entree / Amplitude_sortie 
    #calcul de l'atténuation
   

    gain_dB = 20*np.log10(np.abs(H_15))
    print('Atténuation = ' , H_15,'dB')

    #calcul du gain en db
    plt.figure(figsize=(8,6))
    for index in np.arange(0,len(Q),1):
        H = -K/ (1+1j*Q[index]*(f0 -1 / f0)) 
        Ha = np.abs(H)
       
        gain_dB = 20*np.log10(np.abs(H))
        plt.semilogx(f0,gain_dB, label='Q'+str(Q[index]))
        #K2 = np.abs(K)*(1/np.sqrt(1+Q[index]*(f0 -(1/f0) )))
        #print (K2)
    print(H_15)
    plt.grid(True)
    plt.xlabel('f0')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.ylim([-40,0])
    plt.show()
    
#passeBas1()



