import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
uns = np.ones(250)
zeros = np.zeros(1000-250)
x = np.hstack((uns,zeros))
harmonique = 20

def carre1():
    dt = 1 / 1000  # Increased sampling rate
    t = np.arange(-9, 9, dt)
    u = np.zeros((20,18000),dtype=complex)
    # création du signal avec scipy.signal
    # f_frequency = 1/T  # Frequency corresponding to the period
    duty_cycle = 0.5  # 50% duty cycle for the entire 4-second period
    x2 = signal.square(0.5 * np.pi *(t+1), duty=0.5)

    # affichage du signal

  
 
    w0 = 2*np.pi/4
    
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
    plt.xlim(-2*np.pi,2*np.pi)
    
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
# plt.plot(t, np.real(u[0,:]))
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
carre2()


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
    for k in range(0,20):
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

Q = [1,2,5,10,20,40]
K = 1
f0 = np.arange(0.5, 2, 0.001)

plt.figure(figsize=(8,6))
for index in np.arange(0,len(Q), 1):
    H = -K/ (1+1j*Q[index]*(f0 -1 / f0)) 
    Ha = np.abs(H)
    plt.semilogx(f0,Ha, label='Q'+str(Q[index]))
plt.grid()
plt.xlabel('f0')
plt.ylabel('gain')
plt.show()

#calcul du gain en db
plt.figure(figsize=(8,6))
for index in np.arange(0,len(Q),1):
    H = -K/ (1+1j*Q[index]*(f0 -1 / f0)) 
    Ha = np.abs(H)
    gain_dB = 20*np.log10(np.abs(H))
    plt.semilogx(f0,gain_dB, label='Q'+str(Q[index]))

plt.grid(True)
plt.xlabel('f0')
plt.ylabel('Gain (dB)')
plt.legend()
plt.show()