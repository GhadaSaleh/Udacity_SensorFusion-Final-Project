# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = params.dt
        a=np.matrix([[1,0,0,dt, 0, 0],
                     [0,1,0, 0,dt, 0],
                     [0,0,1, 0, 0,dt],
                     [0,0,0, 1, 0, 0],
                     [0,0,0, 0, 1, 0],
                     [0,0,0, 0, 0, 1]])
        return a      
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############

        dt_cube = params.dt *params.dt*params.dt
        dt_square = params.dt *params.dt
        a11 = dt_cube/3
        a14 = dt_square / 2
        a22 = dt_cube / 3
        a25 = dt_square / 2
        a33 = dt_cube / 3
        a36 = dt_square / 2
        a44 = params.dt
        a55 = params.dt
        a66 = params.dt
        a =params.q * np.matrix([[a11, 0  , 0  , a14, 0  , 0  ],
                                 [0  , a22, 0  , 0  , a25, 0  ],
                                 [0  , 0  , a33, 0  , 0  , a36],
                                 [a14, 0  , 0  , a44, 0  , 0  ],
                                 [0  , a25, 0  , 0  , a55, 0  ],
                                 [0  , 0  , a36, 0  , 0  , a66]])
        return a
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        track.set_x(self.F()*track.x)
        track.set_P(self.Q()+self.F()*track.P*self.F().T)
        ############
        #pass      
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        gama= self.gamma(track,meas)

        P = track.P
        H = meas.sensor.get_H(track.x)
        S = self.S(track,meas,H)
        K = P * H.T * np.linalg.inv(S.astype(float))
        x_new= track.x+K * gama
        I = np.matrix(np.identity(params.dim_state))
        P_new = (I-K*H)*P

        track.set_x(x_new)
        track.set_P(P_new)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        return meas.z -meas.sensor.get_hx(track.x)
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        return (meas.R+H*track.P*H.T)
        ############
        # END student code
        ############ 