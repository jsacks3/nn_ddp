# Module: dynamics.py
# Created on: 4/25/17
# Author: Jake Sacks

import config
import numpy as np

def nonlinear(x, u):
    x3_squared = x[3]**2
    sin_x2 = np.sin(x[2])
    cos_x2 = np.cos(x[2])
    c = 1/(config.m_c + config.m_p*sin_x2**2)

    dx = np.zeros([4,1])
    dx[0] = x[1]
    dx[1] = c*( u - config.m_p*sin_x2 * (config.l * x3_squared - config.g * cos_x2) )
    dx[2] = x[3]
    dx[3] = c/config.l * (u*cos_x2 - config.m_p * config.l * x3_squared * cos_x2 * sin_x2 \
            + (config.m_p + config.m_c) * config.g * sin_x2)

    return dx

def linearized(x, u):
    delta = 0.001
    f_x = np.matrix(np.zeros([config.num_states, config.num_states]))
    f_u = np.matrix(np.zeros([config.num_states, 1]))

    # numerical jacobian wrt x
    x_peturb2 = np.copy(x)
    x_peturb3 = np.copy(x)
    x_peturb2[2] -= delta
    x_peturb3[3] -= delta

    dx = nonlinear(x, u)
    dx_peturb_x2 = nonlinear(x_peturb2, u)
    dx_peturb_x3 = nonlinear(x_peturb3, u)

    f_x[0,1] = 1
    f_x[2,3] = 1
    f_x[1,2] = (dx[1] - dx_peturb_x2[1])/delta
    f_x[1,3] = (dx[1] - dx_peturb_x3[1])/delta
    f_x[3,2] = (dx[3] - dx_peturb_x2[3])/delta
    f_x[3,3] = (dx[3] - dx_peturb_x3[3])/delta

    # numerical jacobian wrt u
    dx_peturb_u = nonlinear(x, u-delta)
    f_u[1] = (dx[1] - dx_peturb_u[1])/delta
    f_u[3] = (dx[3] - dx_peturb_u[3])/delta

    return(f_x, f_u)

