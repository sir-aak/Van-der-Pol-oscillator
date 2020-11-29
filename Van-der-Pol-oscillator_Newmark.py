import numpy as np
import matplotlib.pyplot as plt 


# returns coefficients for 2nd order ode A*ddx + D*dx + S*x = 0
def vanderpol (X, t):
    
    a  = 1
    k  = 1
    w0 = 1
    
    x, dx, ddx = X
    
    A = 1.0
    D = k * (np.power(x, 2) - a)
    S = np.power(w0, 2)
    
    return (A, D, S)


# method for solving 2nd order odes
def Newmark (y, dy, ddy, A, D, S, r, dt, beta, gamma):
    
    y_star  = y + dy * dt + (0.5 - beta) * ddy * np.power(dt, 2)
    dy_star = dy + (1 - gamma) * ddy * dt
    
    ddy = -(D * dy + S * y) / A
    y   = y_star + beta * np.power(dt, 2) * ddy
    dy  = dy_star + gamma * dt * ddy
    
    return (y, dy, ddy)


def numInt ():
    
    Ttrans = 0.0
    Teval  = 50.0
    dt     = 1e-3
    
    nSteps = int((Ttrans + Teval) / dt)
    
    x   = 0.1
    dx  = 0.0
    ddx = 0.0
    
    t   = 0.0
    
    A, D, S   = vanderpol([x, dx, ddx], t)
    r         = 0.0
    
    sol       = np.zeros((2, nSteps))
    sol[:, 0] = np.array([x, dx])
    
    time      = np.zeros(nSteps)
    time[0]   = t
    
    for i in range(1, nSteps):
        time[i]    = i * dt
        t          = time[i]
        x, dx, ddx = Newmark(x, dx, ddx, A, D, S, r, dt, 0.25, 0.5)
        A, D, S    = vanderpol([x, dx, ddx], t)
        sol[:, i]  = np.array([x, dx])
    
    X = sol[0, :]
    Y = sol[1, :]
    
    return (time, X, Y)


def view():
    
    time, X, Y = numInt()
    
    fig = plt.figure()
    fig.suptitle("Van-der-Pol-oscillator", fontsize=14)
    fig.set_size_inches(14, 6)
    fig.subplots_adjust(wspace=0.3)
    
    plt.subplot(1, 2, 1)
    plt.plot(time, X, label=r"$x(t)$")
    plt.plot(time, Y, label=r"$y(t)$")
    plt.legend(loc="upper left")
    plt.title("time series")
    plt.xlabel(r"time $t$")
    plt.grid(color="lightgray")

    plt.subplot(1, 2, 2)
    plt.plot(X, Y, color="black")
    plt.plot(X[0], Y[0], color="red", marker="o", markersize=6)
    plt.title("phase space")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid(color="lightgray")


view()
