# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 23:19:35 2020

@author: kylej
"""

import numpy as np
import pandas as pd
from sympy import Symbol, sympify
import sympy as sp
import scipy.optimize
from symfit import Fit,parameters,variables,GreaterThan,LessThan,Equality,Parameter
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.integrate import quad
import time

class Fourier :
    """
    Fits data with a Fourier Series, hardcoded for 4 features currently
    """
    def __init__(self, input_data, z_label=None, revenue_goal=None, freq1=None, freq2=None, freq3=None, freq4=None, number_of_terms=5):
        #if isinstance(file_name, str):
            #df = pd.read_excel(file_name)
        if isinstance(input_data, pd.DataFrame):
            input_data.columns = ['x', 'x2', 'x3', 'x4', f'{z_label}']
            zdata = input_data[f'{z_label}'].values
            xdata = input_data['x'].values
            x2data = input_data['x2'].values
            x3data = input_data['x3'].values
            x4data = input_data['x4'].values
        elif isinstance(input_data, list):
            x2data = np.asarray(input_data)
        self.z = zdata
        self.x2 = x2data
        self.x = xdata
        self.x3 = x3data
        self.x4 = x4data
        self.label = z_label
        self.revenueGoal = revenue_goal
        self.n = number_of_terms
        if freq1 is not None:
            self.w1 = freq1
        if freq2 is not None:
            self.w2 = freq2
        if freq3 is not None:
            self.w3 = freq1
        if freq4 is not None:
            self.w4 = freq2
        else:
            self.w1 = Parameter('w1', value=0.5)
            self.w2 = Parameter('w2', value=0.5)
            self.w3 = Parameter('w3', value=0.5)
            self.w4 = Parameter('w4', value=0.5)


    def fourier(self):
        n = self.n
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        w4 = self.w4
        lst = range(n+1)
        self.a_n = parameters(','.join(['a{}'.format(i) for i in lst]))
        self.b_n = parameters(','.join(['b{}'.format(i) for i in lst]))
        self.c_n = parameters(','.join(['c{}'.format(i) for i in lst]))
        self.d_n = parameters(','.join(['d{}'.format(i) for i in lst]))
        self.e_n = parameters(','.join(['e{}'.format(i) for i in lst]))
        self.f_n = parameters(','.join(['f{}'.format(i) for i in lst]))        
        self.coeff = self.a_n + self.b_n
        # self.eqn = sum([i * sp.cos(k * w * Symbol('x')) + j * sp.sin(k * w * Symbol('x')) + q * sp.cos(k * w * Symbol('y')) + r * sp.sin(k * w * Symbol('y')) for k, (i, j, q, r) in enumerate(zip(self.a_n, self.b_n, self.c_n, self.d_n))])
        # self.eqn = sum([i * sp.cos(k * w1 * Symbol('x')) + j * sp.cos(k * w2 * Symbol('y')) + q * sp.cos(k * w1 * Symbol('x') + k * w2 * Symbol('y')) + r * sp.cos(k * w1 * Symbol('x') - k * w2 * Symbol('y')) for k, (i, j, q, r) in enumerate(zip(self.a_n, self.b_n, self.c_n, self.d_n))])
        self.eqn = (sum([i * sp.cos(k * w1 * Symbol('x')) + j * sp.cos(k * w2 * Symbol('x2')) 
                         + g * sp.cos(k * w3 * Symbol('x3')) + h * sp.cos(k * w4 * Symbol('x4')) 
                         + q * sp.cos(k * w1 * Symbol('x') + k * w2 * Symbol('x2') + k * w3 * Symbol('x3') + k * w4 * Symbol('x4')) 
                         + r * sp.cos(k * w1 * Symbol('x') - k * w2 * Symbol('x2') - k * w3 * Symbol('x3') - k * w4 * Symbol('x4')) 
                         for k, (i, j, q, r, g, h) in enumerate(zip(self.a_n, self.b_n, self.c_n, self.d_n, self.e_n, self.f_n))])
                    )
        # self.eqn = sum([i * sp.cos(k * w1 * Symbol('x')) + j * sp.cos(k * w2 * Symbol('y')) + q * sp.cos(k * w1 * Symbol('x')) * sp.cos(k * w2 * Symbol('y')) for k, (i, j, q) in enumerate(zip(self.a_n, self.b_n, self.c_n))])
        ##self.eqn = sum([i * sp.cos(k * w1 * Symbol('x')) + j * sp.cos(k * w2 * (Symbol('y'))) for k, (i, j) in enumerate(zip(self.a_n, self.b_n))])
        print(self.eqn)
        return self.eqn


    def fit(self):
        x, x3, x4, x2, z = variables('x, x3, x4, x2, z')
        model_dict = {z: self.fourier()}
        self.ffit = Fit(model_dict, x=self.x, x2=self.x2, x3=self.x3, x4=self.x4, z=self.z)
        self.fit_result = self.ffit.execute()
        self.orderedDict = self.fit_result.params
        print(self.fit_result.params)
        return self.fit_result.params

    def fitFunc(self):
        self.fiteqn = self.eqn
        for k, v in self.orderedDict.items():
            self.fiteqn = self.fiteqn.subs(Parameter('{}'.format(k)), self.orderedDict[k])
        print(self.fiteqn)
        return self.fiteqn

    def fFunc(self, x, x2, x3, x4):
        strfunc = str(self.fiteqn.subs([(Symbol('x'), x), (Symbol('x2'), x2), (Symbol('x3'), x3), (Symbol('x4'), x4)]))
        strfunc = strfunc.replace('sin','np.sin').replace('cos','np.cos')
        return eval(strfunc)

    def fitPlot(self,plot_data=True,color='red'):
        if plot_data == True :
            plt.plot(self.x, self.x2, lw=3, alpha=0.7, label=self.label, color=color)  # plots line that is being fit
        plt.plot(self.x, self.ffit.model(self.x, **self.fit_result.params).z, color='red', ls='--',label='_nolegend_')

    def predict(self, data, label, n_label):
        pred_list = []
        df_label = pd.DataFrame(list(range(0, n_label, 1)), columns = ['cat'])
        
        for index, row in data.iterrows(): 
            prediction = self.fFunc(row[0],row[1],row[2],row[3])
            pred_list.append(prediction)
        data['raw_output'] = pred_list
        data['prediction'] = data['raw_output'].apply(lambda x: abs(df_label['cat'] - x).idxmin())
        return data
