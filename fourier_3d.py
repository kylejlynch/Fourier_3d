# Imports
import numpy as np
import pandas as pd
from sympy import Symbol, sympify
import sympy as sp
import scipy.optimize
from symfit import Fit,parameters,variables,GreaterThan,LessThan,Equality,Parameter
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Fourier :
    """
    Fits data with a Fourier Series
    
    Example use:
    
    ff = Fourier(file_name='3d_data.xlsx', z_label='z', freq1=0.25, freq2=0.15, number_of_terms=5)
    fit_result = ff.fit()
    ff.fitFunc()
    """
    def __init__(self, file_name, z_label=None, revenue_goal=None, freq1=None, freq2=None, number_of_terms=5):
        if isinstance(file_name, str):
            df = pd.read_excel(file_name)
            zdata = df['{}'.format(z_label)].values
            xdata = df['x'].values
            ydata = df['y'].values
        elif isinstance(file_name, list):
            ydata = np.asarray(file_name)
        self.z = zdata
        self.y = ydata
        self.x = xdata
        self.label = z_label
        self.revenueGoal = revenue_goal
        self.n = number_of_terms
        if freq1 is not None:
            self.w1 = freq1
        if freq2 is not None:
            self.w2 = freq2
        else:
            self.w1 = Parameter('w1', value=0.5)
            self.w2 = Parameter('w2', value=0.5)

    def fourier(self):
        n = self.n
        w1 = self.w1
        w2 = self.w2
        lst = range(n+1)
        self.a_n = parameters(','.join(['a{}'.format(i) for i in lst]))
        self.b_n = parameters(','.join(['b{}'.format(i) for i in lst]))
        self.c_n = parameters(','.join(['c{}'.format(i) for i in lst]))
        self.d_n = parameters(','.join(['d{}'.format(i) for i in lst]))
        self.eqn = sum([i * sp.cos(k * w1 * Symbol('x')) + j * sp.cos(k * w2 * Symbol('y')) + q * sp.cos(k * w1 * Symbol('x') + k * w2 * Symbol('y')) + r * sp.cos(k * w1 * Symbol('x') - k * w2 * Symbol('y')) for k, (i, j, q, r) in enumerate(zip(self.a_n, self.b_n, self.c_n, self.d_n))])
        #print(self.eqn)
        return self.eqn

    def fit(self):
        x, y, z = variables('x, y, z')
        model_dict = {z: self.fourier()}
        self.ffit = Fit(model_dict, x=self.x, y=self.y, z=self.z)
        self.fit_result = self.ffit.execute()
        self.orderedDict = self.fit_result.params
        #print(self.fit_result.params)
        return self.fit_result.params

    def fitFunc(self):
        self.fiteqn = self.eqn
        for k, v in self.orderedDict.items():
            self.fiteqn = self.fiteqn.subs(Parameter('{}'.format(k)), self.orderedDict[k])
        #print(self.fiteqn) 
        return self.fiteqn
    
    def fFunc(self, x, y):
        strfunc = str(self.fiteqn.subs([(Symbol('x'), x), (Symbol('y'), y)]))
        strfunc = strfunc.replace('sin','np.sin').replace('cos','np.cos')
        return eval(strfunc)
