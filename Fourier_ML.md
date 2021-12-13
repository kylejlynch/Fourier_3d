<h2><p align="center">Using Multi-Dimensional Fourier Series for Machine Learning -- Iris Dataset</p></h2>
<p align="center">
    By<br>
    Kyle Lynch<br>
    2021-12-13<br>
</p>


A Fourier series is an expansion of a periodic function in terms of an infinite sum of sines and cosines. Please see my other works on [Fourier Forecasting](https://github.com/kylejlynch/Fourier_Forecasting), and [3D Fourier series fitting (with Newton-Raphson approximation)](https://github.com/kylejlynch/Fourier_3d), for more information, or check out the [Wikipedia](https://en.wikipedia.org/wiki/Fourier_series). In this notebook we'll use a multi-dimensional Fourier series on the popular Iris data set. The data set contains three classes of 50 instances each, where each class refers to a type of iris plant. As a quick demonstration of how Fourier series can fit unique sets of data, here is an example of how Fourier series can fit a 2D and 3D square wave.

![2d_3d_square](https://user-images.githubusercontent.com/36255172/145768903-8434d9a4-a423-4373-861c-e968a89475f2.png)

Unfortunately we cannot view the five dimensions needed for the Iris data set in this same way. Now let's load the Fourier class.


```python
# Imports
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
```


```python
class Fourier :
    """
    Fits data with a Fourier Series
    """
    def __init__(self, input_data, z_label=None, revenue_goal=None, freq1=None, freq2=None, freq3=None, freq4=None, number_of_terms=5):
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
        self.eqn = (sum([i * sp.cos(k * w1 * Symbol('x')) + j * sp.cos(k * w2 * Symbol('x2')) 
                         + g * sp.cos(k * w3 * Symbol('x3')) + h * sp.cos(k * w4 * Symbol('x4')) 
                         + q * sp.cos(k * w1 * Symbol('x') + k * w2 * Symbol('x2') + k * w3 * Symbol('x3') + k * w4 * Symbol('x4')) 
                         + r * sp.cos(k * w1 * Symbol('x') - k * w2 * Symbol('x2') - k * w3 * Symbol('x3') - k * w4 * Symbol('x4')) 
                         for k, (i, j, q, r, g, h) in enumerate(zip(self.a_n, self.b_n, self.c_n, self.d_n, self.e_n, self.f_n))])
                    )
        #print(self.eqn)
        return self.eqn


    def fit(self):
        x, x3, x4, x2, z = variables('x, x3, x4, x2, z')
        model_dict = {z: self.fourier()}
        self.ffit = Fit(model_dict, x=self.x, x2=self.x2, x3=self.x3, x4=self.x4, z=self.z)
        self.fit_result = self.ffit.execute()
        self.orderedDict = self.fit_result.params
        #print(self.fit_result.params)
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
```


```python
# Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
```


```python
pd.set_option('max_columns', None)
data = load_iris()
iris_data = pd.DataFrame(data.data,columns=data.feature_names)
iris_data['target'] = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(iris_data.drop('target',axis=1), iris_data['target'], test_size=0.30, random_state=42)

# The way I wrote the code requires that train and test be concatenated
iris_train = pd.concat([X_train, y_train], axis=1)
iris_test = pd.concat([X_test, y_test], axis=1)
iris_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>6.1</td>
      <td>2.8</td>
      <td>4.7</td>
      <td>1.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5.7</td>
      <td>3.8</td>
      <td>1.7</td>
      <td>0.3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>7.7</td>
      <td>2.6</td>
      <td>6.9</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>78</th>
      <td>6.0</td>
      <td>2.9</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>6.8</td>
      <td>2.8</td>
      <td>4.8</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
ff = Fourier(input_data=iris_train, z_label='target', number_of_terms=5)
fit_result = ff.fit()
ff.fitFunc()
```

    11661.6837102083*cos(0.520211768495524*x) + 7046.6563948983*cos(1.04042353699105*x) + 2898.66455451792*cos(1.56063530548657*x) + 731.932920695947*cos(2.0808470739821*x) + 85.7587685793276*cos(2.60105884247762*x) + 0.73364614798943*cos(1.30730866564737*x2) + 0.774974407219831*cos(2.61461733129474*x2) + 0.484512923262722*cos(3.9219259969421*x2) + 0.396557683594972*cos(5.22923466258947*x2) + 0.182396528570974*cos(6.53654332823684*x2) + 0.183443547681612*cos(0.986429337887414*x3) + 0.122264900517642*cos(1.97285867577483*x3) - 0.0818435823173503*cos(2.95928801366224*x3) - 0.034717856594977*cos(3.94571735154965*x3) - 0.0894377186961083*cos(4.93214668943707*x3) - 2.42768007197202*cos(0.805222848474826*x4) + 0.950847469165734*cos(1.61044569694965*x4) - 0.905736326293015*cos(2.41566854542448*x4) + 0.561409386597049*cos(3.2208913938993*x4) - 0.250347408402404*cos(4.02611424237413*x4) - 0.0300688403833114*cos(-2.60105884247762*x + 6.53654332823684*x2 + 4.93214668943707*x3 + 4.02611424237413*x4) + 0.0395687111918803*cos(-2.0808470739821*x + 5.22923466258947*x2 + 3.94571735154965*x3 + 3.2208913938993*x4) - 0.0287881038364621*cos(-1.56063530548657*x + 3.9219259969421*x2 + 2.95928801366224*x3 + 2.41566854542448*x4) + 0.0533988982735975*cos(-1.04042353699105*x + 2.61461733129474*x2 + 1.97285867577483*x3 + 1.61044569694965*x4) + 0.124934961764972*cos(-0.520211768495524*x + 1.30730866564737*x2 + 0.986429337887414*x3 + 0.805222848474826*x4) - 0.111227830313919*cos(0.520211768495524*x + 1.30730866564737*x2 + 0.986429337887414*x3 + 0.805222848474826*x4) - 0.0283031095375071*cos(1.04042353699105*x + 2.61461733129474*x2 + 1.97285867577483*x3 + 1.61044569694965*x4) - 0.0120799410070928*cos(1.56063530548657*x + 3.9219259969421*x2 + 2.95928801366224*x3 + 2.41566854542448*x4) - 0.0453116005919625*cos(2.0808470739821*x + 5.22923466258947*x2 + 3.94571735154965*x3 + 3.2208913938993*x4) - 0.0801786043679045*cos(2.60105884247762*x + 6.53654332823684*x2 + 4.93214668943707*x3 + 4.02611424237413*x4) + 6870.25640964563
    





Above is the equation for our fit!
Now that our model is trained, let's try it on the test set!


```python
pred = ff.predict(iris_test, 'target', 3) # 3 corresponds to the number of labels (target)
pred.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
      <th>raw_output</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>6.1</td>
      <td>2.8</td>
      <td>4.7</td>
      <td>1.2</td>
      <td>1</td>
      <td>0.917022</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5.7</td>
      <td>3.8</td>
      <td>1.7</td>
      <td>0.3</td>
      <td>0</td>
      <td>0.001808</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>7.7</td>
      <td>2.6</td>
      <td>6.9</td>
      <td>2.3</td>
      <td>2</td>
      <td>2.281772</td>
      <td>2</td>
    </tr>
    <tr>
      <th>78</th>
      <td>6.0</td>
      <td>2.9</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>1</td>
      <td>1.175344</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>6.8</td>
      <td>2.8</td>
      <td>4.8</td>
      <td>1.4</td>
      <td>1</td>
      <td>1.155445</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>5.4</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>0</td>
      <td>0.056897</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>5.6</td>
      <td>2.9</td>
      <td>3.6</td>
      <td>1.3</td>
      <td>1</td>
      <td>0.994164</td>
      <td>1</td>
    </tr>
    <tr>
      <th>141</th>
      <td>6.9</td>
      <td>3.1</td>
      <td>5.1</td>
      <td>2.3</td>
      <td>2</td>
      <td>1.789852</td>
      <td>2</td>
    </tr>
    <tr>
      <th>68</th>
      <td>6.2</td>
      <td>2.2</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>1</td>
      <td>1.476185</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>3.9</td>
      <td>1.2</td>
      <td>1</td>
      <td>0.922800</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



In the above output, 'target' is the label column which came from the Iris dataset. 'Raw_output' corresponds to the actual value the model is predicting (regression value), and 'prediction' is which target value the regression value is in closest proximity to (the predicted class). Now let's check our accuracy.


```python
print('accuracy:', 1 - len(pred[~(pred['target']==pred['prediction'])]) / len(pred))
print('count of test set:', len(iris_test))
```

    accuracy: 0.9777777777777777
    count of test set: 45
    

Not bad, looks like we got 1 incorrect out of 45. Let's see which one.


```python
pred[~(pred['target']==pred['prediction'])]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
      <th>raw_output</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>5.7</td>
      <td>4.4</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>0</td>
      <td>1.366737</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



This concludes the notebook. This function is hardcoded to work with data sets with four features only. I am currently working on another version which can take avariable number of inputs.
