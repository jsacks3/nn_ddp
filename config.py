# Module: config.py
# Created on: 4/25/17
# Author: Jake Sacks
import numpy as np

def init():

    # DDP parameters
    global gamma, dt, tf, N
    gamma = 0.5 # scaling parameter
    dt = 0.01 # time step
    tf = 100 # window size
    N = 10 # number of DDP iterations

    # system dynamics
    global m_p, m_c, l, g
    global num_states, num_inputs
    m_p = 1.0 # mass of pendulum
    m_c = 10.1 # mass of cart pole
    l = 0.5 # length of pole
    g = 9.81 # gravitational constant
    num_states = 4 # number of system states
    num_inputs = 1 # number of system controls

    # cost function parameters
    global Q_f, Q, R

    # terminal cost
    Q_f = np.matrix(np.zeros([num_states, num_states]))
    Q_f[0,0] = 0
    Q_f[1,1] = 500
    Q_f[2,2] = 5000
    Q_f[3,3] = 5000

    # running cost weight on state
    Q = np.matrix(np.zeros([num_states, num_states]))
    Q[0,0] = 0
    Q[1,1] = 10
    Q[2,2] = 100
    Q[3,3] = 0

    # running cost weight on control
    R = 0.001