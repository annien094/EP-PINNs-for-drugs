#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:49:44 2023

@author: mac
"""

import scipy.io
import deepxde as dde
from deepxde.backend import tf  # version 2.4.1
import numpy as np
import math

class system_dynamics():

    def __init__(self):

        ## PDE Parameters
        self.a = 0.01
        self.b = 0.15
        self.D = 0.1
        self.k = 10
#        self.mu_1 = 0.2
#        self.mu_2 = 0.3
#        self.epsilon = 0.002

        ## Geometry Parameters
        self.min_x = 0.1
        self.max_x = 10
#        self.min_y = 0.1
#        self.max_y = 10
        self.min_t = 1
        self.max_t = 100
        self.spacing = 0.1

        ## FentonKarma additions
        #self.BCL=100
        #self.ncyc=1
        #self.extra=0

        self.uv=0.160 # uc for v
        self.uw=0.160 # uc for w
        self.uu=0.160 # uc for u
        self.uvsi=0.040 # uv
        self.ucsi=0.85 # uc_si
        #self.tauv=0
        self.taud=0.125#0.03#0.05569#0.501#0.125 # tau_d
        self.tauv2=60.0 # tauv2-
        self.tauv1=82.5 # tauv1-
        self.tauvplus=5.75 # tauv+
        #self.tauo=10  #tauo
        self.tauo=10#50.76#28.43#38.85#17.6#598
        self.tauwminus=400.0 # tauw-
        self.tauwplus=300.0 # tauw+
        self.taur=100#92.71#79.73#98.11#124#127.0 #120.0 #70
        self.tausi=300#35.15#29.40#0.24277#4.59#114.0 # tausi



    def generate_data(self, file_name, dim):

        data = scipy.io.loadmat(file_name)
        if dim == 1:
            t, x, usav, w = data["t"], data["x"], data["Vsav"], data["Wsav"]
            X, T = np.meshgrid(x, t)
        elif dim == 2:
            t, x, y, usav = data["t"], data["X"], data["y"], data["usav"]
            X, T, Y = np.meshgrid(x,t,y)
            Y = Y.reshape(-1, 1)
        else:
            raise ValueError('Dimesion value argument has to be either 1 or 2')

        self.max_t = np.max(t)
        self.max_x = np.max(x)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        U = usav.reshape(-1, 1)
        W = w.reshape(-1, 1)

        if dim == 1:
            return np.hstack((X, T)), U, W
        #return np.hstack((X, Y, T)), U  # add if statement for dim = 2

    def geometry_time(self, dim):
        if dim == 1:
            geom = dde.geometry.Interval(self.min_x, self.max_x)
            timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        elif dim == 2:
            geom = dde.geometry.Rectangle([self.min_x,self.min_y], [self.max_x,self.max_y])
            timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        else:
            raise ValueError('Dimesion value argument has to be either 1 or 2')
        return geomtime

    def params_to_inverse(self,args_param):

        params = []
        if not args_param:
            return self.taud, self.taur, self.tauo, self.D, params
        ## If inverse:
        ## The tf.variables are initialized with a positive scalar, relatively close to their ground truth values
        if 'taud' in args_param:
            #MATLAB values: taud=0.125 , taur=70 , tauo=32.5 , d=0.1,  tausi=114

            #self.taud = tf.math.exp(tf.Variable(-0.69897)) #initiasilised to 0.2
            print("Estimatig taud")
            #self.taud = tf.math.exp(tf.Variable(-0.69314718056)) #0.501
            #self.taud = tf.math.exp(tf.Variable(-2.88795468138)) #0.06
            #self.taud = tf.math.exp(tf.Variable(-1.897119985))  #initiasilised to 0.15
            #self.taud = tf.math.exp(tf.Variable(-2.353878387))  #initiasilised to 0.095
            #self.taud = tf.math.exp(tf.Variable(-2.88204650915017)) #0.05602
            #self.taud = tf.math.exp(tf.Variable(-3.50655789731998)) #0.03
            self.taud = tf.math.exp(tf.Variable(-2.07944154167984)) #0.125
            params.append(self.taud)
        if 'taur' in args_param:
            #self.taur = tf.math.exp(tf.Variable(1.69897))
            print("Estimatig taur")
            #self.taur = tf.math.exp(tf.Variable(4.787491743))  #initialised to 120 (originally 70)
            #self.taur = tf.math.exp(tf.Variable(4.84418708646)) #initialise to 127
            #self.taur = tf.math.exp(tf.Variable(4.82028156561)) #124
            #self.taur = tf.math.exp(tf.Variable(4.58608929818)) #98.11
            #self.taur = tf.math.exp(tf.Variable(4.44434910844126)) #85.14444
            #self.taur = tf.math.exp(tf.Variable(4.3786459265144)) #79.73
            #self.taur = tf.math.exp(tf.Variable(4.52947634161889)) #92.71
            self.taur = tf.math.exp(tf.Variable(4.60517018598809)) #100
            params.append(self.taur)
        if 'tauo' in args_param:
            #self.tauo = tf.math.exp(tf.Variable(1.602))
            print("Estimatig tauo")
            #self.tauo = tf.math.exp(tf.Variable(2.30258509299))  #initialised to 10 (originally 31, exp(3.433987204))
            #self.tauo = tf.math.exp(tf.Variable(6.39359075395)) #598
            #self.tauo = tf.math.exp(tf.Variable(2.86789890204)) #17.6
            #self.tauo = tf.math.exp(tf.Variable(3.65970807681)) #38.85
            #self.tauo = tf.math.exp(tf.Variable(3.98568639305453)) #53.82222
            #self.tauo = tf.math.exp(tf.Variable(3.34744492566291)) #28.43
            #self.tauo = tf.math.exp(tf.Variable(3.92710864284619)) #50.76
            self.tauo = tf.math.exp(tf.Variable(2.30258509299405)) #10
            params.append(self.tauo)
        if 'tausi' in args_param:
            print("Estimatig tausi")
            #self.tausi = tf.math.exp(tf.Variable(1.52388002407))  #initialised to 4.59
            #self.tausi = tf.math.exp(tf.Variable(-1.41564078592)) #0.24
            #self.tausi = tf.math.exp(tf.Variable(-1.41239196024176)) #0.24356
            #self.tausi = tf.math.exp(tf.Variable(3.38)) #29.40
            #self.tausi = tf.math.exp(tf.Variable(3.55962461825667)) #35.15
            self.tausi = tf.math.exp(tf.Variable(5.7037824746562)) #300
            params.append(self.tausi)
        if 'd' in args_param:
            #self.D = tf.math.exp(tf.Variable(-1.6))
            print("Estimatig D")
            #self.D = tf.math.exp(tf.Variable(-2.302585093))  #initialised to 0.1
            self.D = tf.math.exp(tf.Variable(-1.897119985))  #initialised to 0.15
            params.append(self.D)
        return params

    def pde_1D(self, x, y):
        u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        dv_dt = dde.grad.jacobian(v, x, i=0, j=1)
        du_dt = dde.grad.jacobian(u, x, i=0, j=1)
        du_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dw_dt = dde.grad.jacobian(w, x, i=0, j=1)


        tauv = tf.cast(tf.math.less_equal(self.uvsi, u),tf.float32)*self.tauv2 + tf.cast(tf.math.less_equal(u, self.uvsi),tf.float32)*self.tauv1
        tauv = tf.cast(tf.math.less_equal(u, self.uv),tf.float32)*tauv + tf.cast(tf.math.less_equal(self.uv, u),tf.float32)*self.tauvplus
        vinf = tf.cast(tf.math.less_equal(u, self.uv),tf.float32)
        Fu = tf.cast(tf.math.less_equal(self.uv,u),tf.float32)*((u-self.uv)*(tf.ones([1],tf.float32)-u))
        Jfi = Fu*(-v) / self.taud  # Fast Inward current
        #v = v + (vinf - v) / tauv*dt_tf  # Update v
        Uu = tf.cast(tf.math.less_equal(self.uu, u),tf.float32) + tf.cast(tf.math.less_equal(u, self.uu),tf.float32)*u
        tauu = tf.cast(tf.math.less_equal(self.uu, u),tf.float32)*self.taur + tf.cast(tf.math.less_equal(u, self.uu),tf.float32)*self.tauo  # old
        #tauu = tf.cast(tf.math.less_equal(self.uu, u),tf.float32)*self.taur + tf.cast(tf.math.less_equal(u, self.uu),tf.float32)*(0.4642857143*self.taur)  # as a function of taur
        #tauu = tf.cast(tf.math.less_equal(self.uu, u),tf.float32)*(2.153846154*self.tauo) + tf.cast(tf.math.less_equal(u, self.uu),tf.float32)*self.tauo  # as a function of tauo
        Jso = Uu/tauu
        #winf = tf.cast(tf.math.less_equal(u, self.uw),tf.float32)
        winf = tf.cast(tf.math.less_equal(u, self.uw),tf.float32) + tf.cast(tf.math.less_equal(self.uw, u),tf.float32)*0
        #tauw = tf.cast(tf.math.less_equal(u, self.uw),tf.float32)*self.tauwminus + tf.cast(tf.math.less_equal(self.tauwminus, u),tf.float32)*self.tauwplus
        tauw = tf.cast(tf.math.less_equal(u, self.uw),tf.float32)*self.tauwminus + tf.cast(tf.math.less_equal(self.uw, u),tf.float32)*self.tauwplus
        Jsi = -w/self.tausi/2*(tf.ones([1],tf.float32) + tf.nn.tanh(self.k*(u-self.ucsi)))


        # start boundary and initial conditions
        #x_space,t_space = x[:, 0:1],x[:, 1:2]
        #t_stim_1 = tf.equal(t_space, 0)
        #t_stim_2 = tf.equal(t_space, int(self.max_t/2))
        #x_stim = tf.less_equal(x_space, 5*self.spacing)

        #first_cond_stim = tf.logical_and(t_stim_1, x_stim)
        #second_cond_stim = tf.logical_and(t_stim_2, x_stim)

        #I_stim = tf.ones_like(x_space)*0.1
        #I_not_stim = tf.ones_like(x_space)*0
        #Istim = tf.where(first_cond_stim, I_stim, I_not_stim)
        #Istim = tf.where(tf.logical_or(first_cond_stim,second_cond_stim),I_stim,I_not_stim)
        # end boundary and initial conditions

        Iion = -(Jfi + Jsi + Jso) #modified

        eq_a = du_dt - (Iion+self.D*du_dxx)
        eq_b = dv_dt - (vinf - v) / tauv
        eq_c = dw_dt - (winf - w) / tauw

        # plot Iion to check it's the same
        return [eq_a, eq_b, eq_c]

    def pde_1D_2cycle(self,x, y):

        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=1)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=1)

        x_space,t_space = x[:, 0:1],x[:, 1:2]
        t_stim_1 = tf.equal(t_space, 0)
        t_stim_2 = tf.equal(t_space, int(self.max_t/2))
        x_stim = tf.less_equal(x_space, 5*self.spacing)

        first_cond_stim = tf.logical_and(t_stim_1, x_stim)
        second_cond_stim = tf.logical_and(t_stim_2, x_stim)

        I_stim = tf.ones_like(x_space)*0.1
        I_not_stim = tf.ones_like(x_space)*0
        Istim = tf.where(tf.logical_or(first_cond_stim,second_cond_stim),I_stim,I_not_stim)
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*dv_dxx + self.k*V*(V-self.a)*(V-1) +W*V -Istim
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]

    def pde_2D(self, x, y):

        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*(dv_dxx + dv_dyy) + self.k*V*(V-self.a)*(V-1) +W*V
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]

    def pde_2D_heter(self, x, y):

        V, W, var = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dy = dde.grad.jacobian(y, x, i=0, j=1)

        ## Heterogeneity
        D_heter = tf.math.sigmoid(var)*0.08+0.02;
        dD_dx = dde.grad.jacobian(D_heter, x, i=0, j=0)
        dD_dy = dde.grad.jacobian(D_heter, x, i=0, j=1)

        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  D_heter*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]

        ## Added code
        # class Pars:   #Python does not support structs so you can create an emty class and define atttributes
        #     pass
        #
        # pars = Pars()
        # pars.name = "homog1noPVcyc"
        # pars.X = 160
        # pars.Y = 160
        # # pars.rad = 8
        # pars.D = np.ones(pars.X,pars.Y)/4
        # pars.dt = 0.010
        # pars.gathert = round(10/pars.dt)
        # pars.nms = 400
        # n = 20
        # pars.nelec = n**2
        #
        # pars.elpos[1,:]=2+np.linspace(pars.X/(n+2), pars.X-pars.X/(n+2), num=n)
        # pars.elpos[2,:]=2+np.linspace(pars.Y/(n+2), pars.Y-pars.Y/(n+2), num=n)
        #
        # pars.pacegeo = np.zeros(pars.X, pars.Y)
        # pars.pacegeo[1:20,:] = 1
        # pars.crossgeo = np.zeros(pars.X, pars.Y)
        # pars.crossgeo[:,round(pars.Y/2):end] = 1
        # pars.crosstime = 102
        # pars.stimdur = 2
        # pars.h = 0.3
        # pars.showms = 10
        #
        # pars.szscreenx = 1183
        # pars.szscreeny = 821
        #
        # pars.diff = 1
        # pars.iscyclic = [0, 0]
        # pars.iso = 1

        #pars.radPV = [10, 12, 8, 14]
        #pars.posPV=[[105, 40],[107, 75], [10, 31], [16, 64]]

        # Add imbinarize section


    def pde_2D_heter_forward(self, x, y):

        V, W, D = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dy = dde.grad.jacobian(y, x, i=0, j=1)

        ## Heterogeneity
        dD_dx = dde.grad.jacobian(D, x, i=0, j=0)
        dD_dy = dde.grad.jacobian(D, x, i=0, j=1)

        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  D*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]

    def IC_func(self,observe_train, v_train):

        T_ic = observe_train[:,-1].reshape(-1,1)
        idx_init = np.where(np.isclose(T_ic,0))[0]  # before:  idx_init = np.where(np.isclose(T_ic,1))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        print("v_init")
        print(v_init)
        print("observe_init")
        print(observe_init)
        return dde.PointSetBC(observe_init,v_init,component=0)

    def BC_func(self,dim, geomtime):
        if dim == 1:
            bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        elif dim == 2:
            bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), self.boundary_func_2d, component=0)
        return bc

    def boundary_func_2d(self,x, on_boundary):
            return on_boundary and ~(x[0:2]==[self.min_x,self.min_y]).all() and  ~(x[0:2]==[self.min_x,self.max_y]).all() and ~(x[0:2]==[self.max_x,self.min_y]).all()  and  ~(x[0:2]==[self.max_x,self.max_y]).all()

    def modify_inv_heter(self, x, y):
        domain_space = x[:,0:2]
        D = tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(domain_space, 60,
                            tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 1, activation=None)
        return tf.concat((y[:,0:2],D), axis=1)

    def modify_heter(self, x, y):

        x_space, y_space = x[:, 0:1], x[:, 1:2]

        x_upper = tf.less_equal(x_space, 54*0.1)
        x_lower = tf.greater(x_space,32*0.1)
        cond_1 = tf.logical_and(x_upper, x_lower)

        y_upper = tf.less_equal(y_space, 54*0.1)
        y_lower = tf.greater(y_space,32*0.1)
        cond_2 = tf.logical_and(y_upper, y_lower)

        D0 = tf.ones_like(x_space)*0.02
        D1 = tf.ones_like(x_space)*0.1
        D = tf.where(tf.logical_and(cond_1, cond_2),D0,D1)
        return tf.concat((y[:,0:2],D), axis=1)