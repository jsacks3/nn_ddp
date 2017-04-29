# Module: ddp.py
# Created on: 4/25/17
# Author: Jake Sacks

import config
import dynamics
import numpy as np
from numpy.linalg import inv

def runningCost(x, u, x_f):
    L = (u.T * config.R * u + (x - x_f).T * config.Q * (x - x_f)) * config.dt
    return L

def runningCostDeriv(x, u, x_f):
    L_u = config.R * u * config.dt
    L_x = config.Q * (x - x_f) * config.dt
    L_xx = config.Q * config.dt
    L_uu = config.R * config.dt
    L_xu = np.zeros([config.num_states, config.num_inputs])
    L_ux = np.zeros([config.num_inputs, config.num_states])
    return (L_u, L_x, L_xx, L_uu, L_xu, L_ux)

def terminalCost(x, x_f):
    phi = ((x - x_f).T * config.Q_f * (x - x_f))
    return phi

def terminalCostDeriv(x, x_f):
    phi_x = config.Q_f * (x - x_f)
    phi_xx = config.Q_f.copy()
    return (phi_x, phi_xx)

def run(x_0, x_f):

    # trajectories
    x_new = np.matrix(np.zeros([config.num_states, config.tf]))
    u_new = np.matrix(np.zeros([config.num_inputs, config.tf-1]))
    x_bar = np.matrix(np.zeros([config.num_states, config.tf]))
    u_bar = np.matrix(np.zeros([config.num_inputs, config.tf-1]))

    # gains
    k = np.matrix(np.zeros([config.num_inputs, config.tf-1]))
    K = np.zeros([config.num_inputs, config.num_states, config.tf - 1])

    # linearization
    I_phi = np.zeros([config.tf, config.num_states, config.num_states])
    B = np.zeros([config.tf, config.num_states, 1])

    # cost per iteration of DDP
    cost_vect = np.zeros(config.N)

    # main DDP loop
    for i in range(0, config.N):
        V = terminalCost(x_bar[:, config.tf-1], x_f)
        (V_x, V_xx) = terminalCostDeriv(x_bar[:, config.tf-1], x_f)

        # backward pass
        for j in range(config.tf-2, -1, -1):
            x = x_bar[:, j]
            u = u_bar[:, j]
            (fx, fu) = dynamics.linearized(x, u)
            I_phi[j] = np.eye(config.num_states, config.num_states) + fx * config.dt
            B[j] = fu * config.dt

            L = runningCost(x, u, x_f)
            (L_u, L_x, L_xx, L_uu, L_xu, L_ux) = runningCostDeriv(x, u, x_f)
            Q_0 = L + V
            Q_x = L_x + np.dot(I_phi[j].T, V_x)
            Q_u = L_u + np.dot(B[j].T, V_x)
            Q_xx = L_xx + np.dot(np.dot(I_phi[j].T, V_xx), I_phi[j])
            Q_uu = L_uu + np.dot(np.dot(B[j].T, V_xx), B[j])
            Q_ux = L_ux + np.dot(np.dot(B[j].T, V_xx), I_phi[j])

            k[:,j] = -inv(Q_uu)*Q_u
            K[:,:,j] = -inv(Q_uu)*Q_ux

            V = Q_0 - 0.5 * k[:, j].T*Q_uu*k[:, j]
            V_x = Q_x - np.dot(K[:,:, j].T, Q_uu) * k[:, j]
            V_xx = Q_xx - np.dot(np.dot(K[:,:, j].T, Q_uu), K[:,:,j])

        # forward pass
        x_new[:,0] = x_0.copy()
        dx = x_new[:,0] - x_bar[:,0]
        cost = 0

        for j in range(0, config.tf-1):

            # compute control input
            du = k[:,j] + np.dot(K[:,:,j], dx)
            u_new[:,j] = u_bar[:,j] + du * config.gamma
            dx = I_phi[j]*dx + B[j]*du

            # simulate system to get new state
            x_dot = dynamics.nonlinear(x_new[:,j], u_new[:,j])
            x_new[:,j+1] = x_new[:,j] + x_dot * config.dt

            # update cost
            cost += runningCost(x_new[:,j], u_new[:,j], x_f)

        # add terminal cost
        cost += terminalCost(x_new[:, config.tf - 1], x_f)
        cost_vect[i] = cost

        # update nominal trajectory
        u_bar = u_new.copy()
        x_bar = x_new.copy()

    return (x_new, u_new, cost_vect)