<h2><p align="center">Using Newton-Raphson Method to Solve 3D Fourier Series Fitted Surfaces</p></h2>
<p align="center">
    By<br>
    Kyle Lynch</center><br>
    2021-12-09<br>
</p>

#### In this notebook I'll discuss how to:
1. Fit 3D data with a Fourier cosine series
2. Use Newton-Raphson method to iteratively solve fitted equation that extracts X and Y values that solve for a given Z

Fourier series are well known for their ability to fit almost any distribution of data. In this work I'll explain how to fit 3-dimensional data with a Fourier cosine series, followed by using an numerical approximation method known as Newton-Raphson method to solve for points on the resulting Fourier surface. I go a little more in depth on the basics of Fourier series and show how to do forecasting with Fourier series [here](https://github.com/kylejlynch/Fourier_Forecasting). I have an additional notebook explaining 2D Newton-Raphson [here](https://github.com/kylejlynch/Newton_Raphson).
As an example to highlight the difference between a 2D and 3D Fourier Series, here I fit both a 2D and 3D square wave (we'll fit the 3D square wave in the code below).


```python
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
```


```python
class Fourier :
    """
    Fits data with a Fourier Series
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
```

#### In order to test this I generated some data in Excel files:
1. A 3D step function (3D block)
2. A 3D Gaussian

Let's take a look at the data


```python
df = pd.read_excel('square_3d.xlsx') 
# 0 value everywhere except in the range from -3 to 3 on both the x and y axes in which z = 5
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-7.25</td>
      <td>-7.25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-7.25</td>
      <td>-7.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-7.25</td>
      <td>-6.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-7.25</td>
      <td>-6.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-7.25</td>
      <td>-6.25</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View of square_3d.xlsx
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(df['x'], df['y'], df['z'])
plt.show()
```


    
![output_7_0](https://user-images.githubusercontent.com/36255172/145622913-3e8ce8df-40ca-4700-9037-3cada7dc55ed.png)
    


#### Now to fit the data above with the Fourier code above. Let's vary the number of terms to see how well it fits the data.


```python
ff = Fourier(file_name='square_3d.xlsx', z_label='z', freq1=0.25, freq2=0.15, number_of_terms=5)
fit_result = ff.fit()
print(ff.fitFunc())
```

    -53.7364005637131*cos(0.25*x) + 38.1994739894838*cos(0.5*x) - 22.1614095100097*cos(0.75*x) + 9.98554579694824*cos(1.0*x) - 3.12572366476257*cos(1.25*x) - 21373.9488810488*cos(0.15*y) + 13506.2170679994*cos(0.3*y) - 6007.11892559254*cos(0.45*y) + 1708.07776850681*cos(0.6*y) - 237.475726815423*cos(0.75*y) + 2.46393341310683*cos(0.25*x - 0.15*y) + 2.47227923640621*cos(0.25*x + 0.15*y) + 0.38205850666991*cos(0.5*x - 0.3*y) + 0.374486259236903*cos(0.5*x + 0.3*y) + 0.615553128663866*cos(0.75*x - 0.45*y) + 0.621917274074822*cos(0.75*x + 0.45*y) - 0.311859521957466*cos(1.0*x - 0.6*y) - 0.316555977810864*cos(1.0*x + 0.6*y) - 0.0507246353099343*cos(1.25*x - 0.75*y) - 0.0479976087057799*cos(1.25*x + 0.75*y) + 12433.6234526802
    


```python
x = np.arange(-7, 7, 0.05)
y = np.arange(-7, 7, 0.05)
X, Y = np.meshgrid(x, y)
Z = ff.fFunc(X,Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.set_zlim(-1.0, 5.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()
```


    
![output_10_0](https://user-images.githubusercontent.com/36255172/145622945-ceeadcee-cf1d-40c9-a66d-da6b69b1843a.png)
    


#### Now let's increase the number of terms


```python
ff = Fourier(file_name='square_3d.xlsx', z_label='z', freq1=0.25, freq2=0.15, number_of_terms=20)
fit_result = ff.fit()
print(ff.fitFunc())
```

    5453.97363194277*cos(0.25*x) + 3096.4157373821*cos(0.5*x) - 10423.7085869859*cos(0.75*x) + 10080.7931168054*cos(1.0*x) - 1941.53683588888*cos(1.25*x) - 8417.57009120421*cos(1.5*x) + 13870.0260060319*cos(1.75*x) - 10589.8815345007*cos(2.0*x) + 172.419140031617*cos(2.25*x) + 12003.6310257316*cos(2.5*x) - 20459.5775693331*cos(2.75*x) + 22588.8053776129*cos(3.0*x) - 19250.3070298191*cos(3.25*x) + 13293.2509217136*cos(3.5*x) - 7521.57371616911*cos(3.75*x) + 3451.82526637215*cos(4.0*x) - 1244.91239685382*cos(4.25*x) + 330.327195096185*cos(4.5*x) - 55.5055593757824*cos(4.75*x) + 3.71888561074001*cos(5.0*x) + 25411.7788254111*cos(0.15*y) + 52544.2219993096*cos(0.3*y) - 9792.55870624065*cos(0.45*y) - 53496.0804232167*cos(0.6*y) + 9025.15827797479*cos(0.75*y) + 54784.4617595346*cos(0.9*y) - 13287.249209991*cos(1.05*y) - 55819.449675962*cos(1.2*y) + 23473.2956825209*cos(1.35*y) + 54151.5263866946*cos(1.5*y) - 42019.6287487197*cos(1.65*y) - 42244.3551605681*cos(1.8*y) + 68880.9238182208*cos(1.95*y) + 493.176188346702*cos(2.1*y) - 80191.987069957*cos(2.25*y) + 96585.3653178768*cos(2.4*y) - 62172.8082993734*cos(2.55*y) + 24473.0177668135*cos(2.7*y) - 5629.17407933511*cos(2.85*y) + 587.841752855767*cos(3.0*y) + 634.382402232455*cos(0.25*x - 0.15*y) + 636.407596037922*cos(0.25*x + 0.15*y) - 153.648618431164*cos(0.5*x - 0.3*y) - 155.638164383097*cos(0.5*x + 0.3*y) + 66.2283412967681*cos(0.75*x - 0.45*y) + 68.1596273749447*cos(0.75*x + 0.45*y) - 35.0680767039013*cos(1.0*x - 0.6*y) - 36.9198705282984*cos(1.0*x + 0.6*y) + 20.5189703481922*cos(1.25*x - 0.75*y) + 22.2726638379278*cos(1.25*x + 0.75*y) - 12.9581416096079*cos(1.5*x - 0.9*y) - 14.5974046647993*cos(1.5*x + 0.9*y) + 8.50345779316019*cos(1.75*x - 1.05*y) + 10.0151762006436*cos(1.75*x + 1.05*y) - 5.61611622219212*cos(2.0*x - 1.2*y) - 6.99058981696564*cos(2.0*x + 1.2*y) + 3.74817805852899*cos(2.25*x - 1.35*y) + 4.97888150417829*cos(2.25*x + 1.35*y) - 2.57272540323416*cos(2.5*x - 1.5*y) - 3.65690015876568*cos(2.5*x + 1.5*y) + 1.69544142485103*cos(2.75*x - 1.65*y) + 2.63380278589726*cos(2.75*x + 1.65*y) - 1.11273823082276*cos(3.0*x - 1.8*y) - 1.90865408395137*cos(3.0*x + 1.8*y) + 0.716915305206564*cos(3.25*x - 1.95*y) + 1.37745972667197*cos(3.25*x + 1.95*y) - 0.431352013281622*cos(3.5*x - 2.1*y) - 0.965309875593292*cos(3.5*x + 2.1*y) + 0.230896790291396*cos(3.75*x - 2.25*y) + 0.6494231768098*cos(3.75*x + 2.25*y) - 0.132586795046948*cos(4.0*x - 2.4*y) - 0.448660900254783*cos(4.0*x + 2.4*y) + 0.0850889575596586*cos(4.25*x - 2.55*y) + 0.311894204153997*cos(4.25*x + 2.55*y) - 0.0316608837401495*cos(4.5*x - 2.7*y) - 0.183768066878881*cos(4.5*x + 2.7*y) + 0.0108551085993822*cos(4.75*x - 2.85*y) + 0.101627612726815*cos(4.75*x + 2.85*y) - 0.00160141840788963*cos(5.0*x - 3.0*y) - 0.0429334528793861*cos(5.0*x + 3.0*y) - 51243.3578917042
    


```python
x = np.arange(-7, 7, 0.05)
y = np.arange(-7, 7, 0.05)
X, Y = np.meshgrid(x, y)
Z = ff.fFunc(X,Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.set_zlim(-1.0, 5.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()
```


    
![output_13_0](https://user-images.githubusercontent.com/36255172/145622963-0351d05d-5904-488a-9dcb-0f23f42b0f86.png)
    


#### As you can see, it fits much better! Now let's take a look at the Gaussian.


```python
df = pd.read_excel('gauss_3d.xlsx') 
plt.rcParams['figure.figsize'] = [8, 8]
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(df['x'], df['y'], df['z'])
plt.show()
```


    
![output_15_0](https://user-images.githubusercontent.com/36255172/145622991-f8ccca70-d01b-4854-b862-a212b868c4ca.png)
    



```python
ff = Fourier(file_name='gauss_3d.xlsx', z_label='z', freq1=0.50, freq2=0.50, number_of_terms=1)
fit_result = ff.fit()
print(ff.fitFunc())
```

    1.06204154270031*cos(0.5*x) + 1.06204154270031*cos(0.5*y) + 0.564645486689948*cos(0.5*x - 0.5*y) + 0.564646557430445*cos(0.5*x + 0.5*y) + 0.998795606530294
    


```python
x = np.arange(-7, 7, 0.05)
y = np.arange(-7, 7, 0.05)
X, Y = np.meshgrid(x, y)
Z = ff.fFunc(X,Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.set_zlim(-1.0, 5.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()
```


    
![output_17_0](https://user-images.githubusercontent.com/36255172/145623016-54bfdb2d-e238-4a80-afe4-959b4d342afd.png)
    


#### As you can see, the gaussian is much easier to fit so it fits well with the number of terms set to 1!

#### Now let's use the Newton-Raphson method to find where a known surface equals a desired value

## 3D Newton-Raphson Method
In another notebook, I explain how to use 2-dimensional Newton's method to find zeros of a function (where y = 0 on an x-y coordinate plane). Let's apply this logic to the 3D surfaces we just generated! But first let's extend what we learned about 2D Newton's method to 3D.

In the 2D version, we learned how to apply Newton's method to numerically solve for the zeros of a function. The concept here is the same, but with the added dimension we have to add some additional logic. Newton's method can only solve for a single point at a time, so we'll have to add an intersecting plane that passes (vertically) through the horizontal x-y plane, as well as through the surface of interest. The code below generates a visualization of a single plane passing through a surface (for demonstration purposes). 


```python
x = np.arange(-7, 7, 0.2)
y = np.arange(-7, 7, 0.2)
X, Y = np.meshgrid(x, y)
Z = ff.fFunc(X,Y)
# Plane
x_p = np.arange(-1, 7, 0.2)
z_p = np.arange(-1, 7, 0.2)
X_p, Z_p = np.meshgrid(x_p, z_p)
Y_p = 2 - X_p



fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,alpha=0.8)
surf2 = ax.plot_surface(X_p, Y_p, Z_p, color='red',alpha=0.3)

ax.set_zlim(0, 5.0)
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()
```


    
![output_22_0](https://user-images.githubusercontent.com/36255172/145623042-7b51bd6f-97c2-4ab8-b1ad-f9b4cd587e6e.png)
    


As described above, the figure shows our surface of interest along with the plane that must be used to intersect with the surface and x-y plane. The point of intersection of the surface, the x-y plane, and the vertical plane is what Newton's method is used to find. We can then change the plane to solve for a new intersection point as displayed below.


```python
x = np.arange(-7, 7, 0.2)
y = np.arange(-7, 7, 0.2)
X, Y = np.meshgrid(x, y)
Z = ff.fFunc(X,Y)


x_p2 = np.arange(-1, 7, 0.2)
z_p2 = np.arange(-1, 7, 0.2)
X_p2, Z_p2 = np.meshgrid(x_p2, z_p2)
Y_p2 = 2 + 1*X_p2


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,alpha=0.8)
surf3 = ax.plot_surface(X_p2, Y_p2, Z_p2, color='red',alpha=0.3)

ax.set_zlim(0, 5.0)
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()
```


    
![output_24_0](https://user-images.githubusercontent.com/36255172/145623063-f5f81395-6706-45e6-bbb1-65d9af141400.png)
    


We can iteratively change the plane to find many intersection points with our surface and the x-y plane. Let's first demonstrate with the gaussian output from the Fourier fit. Let's say with want to find the points on the surface at the cross-section at z = 3.5


```python
# Imports
from sympy import sin, cos, symbols, lambdify, diff, exp
import numpy as np
import random
```


```python
x, y = symbols('x y')
gaussian = 1.06204154270031*cos(0.5*x) + 1.06204154270031*cos(0.5*y) + 0.564645486689948*cos(0.5*x - 0.5*y) + 0.564646557430445*cos(0.5*x + 0.5*y) + 0.998795606530294
f2 = -3.5
f1 = gaussian + f2
```


```python
def newton3d(xn,yn,f1,f2):
    X = np.array([[xn],[yn]])
    f = np.array([[f1(xn,yn)],[f2(xn,yn)]])
    J = np.array([[dx_f1(xn,yn), dy_f1(xn,yn)],[dx_f2(xn),dy_f2(yn)]])
    J = np.linalg.inv(J)
    return X - np.dot(J,f)


dx_f1 = diff(f1, x)
dy_f1 = diff(f1, y)

f1 = lambdify([x,y], f1)
dx_f1 = lambdify([x,y], dx_f1)
dy_f1 = lambdify([x,y], dy_f1)

fwd = list(np.linspace(9999,99999,100))
rev = fwd[::-1]
tol = 0.000001 # threshold for convergence (difference between iterations must be below this value)
converge_dict = {'nx':[],'ny':[]}
for k,l in zip(rev, fwd):
    for f2 in [k*x-l*y,k*x+l*y]:
        dx_f2 = diff(f2, x)
        dy_f2 = diff(f2, y)
        
        f2 = lambdify([x,y], f2)
        dx_f2 = lambdify(x, dx_f2)
        dy_f2 = lambdify(y, dy_f2)
        
        # i = 0; n_x=0.50; n_y=0.501; prev_x=-9999; prev_y=-9999
        i = 0; n_x=round(random.uniform(-1, 1), 3); n_y=round(random.uniform(-1, 1), 3); prev_x=-9999; prev_y=-9999
        round(random.uniform(-1, 1), 3)
        while i < 10:
            n = newton3d(n_x, n_y, f1, f2)
            n_x=n[0][0]; n_y=n[1][0]
            if i == 9:
                if (abs(n_x - prev_x) < tol) and (abs(n_y - prev_y) < tol):
                    converge = 'Y'
                    converge_dict['nx'].append(n_x)
                    converge_dict['ny'].append(n_y)
                else:
                    converge = 'N'
            i += 1
            prev_x=n_x; prev_y=n_y

```


```python
converge_df = pd.DataFrame(converge_dict) # Collect the points that converged
```


```python
ax1 = converge_df.plot.scatter(x='nx',y='ny')
ax1.set_xlim(-10,10)
ax1.set_ylim(-10,10)
# Here I manually set my limits to the area of interest. Fourier fits can get messy outside of the area of interest
# However, there are points the do meet the criteria (converge at intersection points outside the area of interest)
```







    
![output_30_1](https://user-images.githubusercontent.com/36255172/145623092-8479cf00-d626-4b3a-bbb7-4b7621d2cc1a.png)
    


#### Now let's try the 3D square wave/block
Again, let's chose z = 3.5 for an interesting cross-section


```python
x, y = symbols('x y')
fourier_square = 5453.97363194277*cos(0.25*x) + 3096.4157373821*cos(0.5*x) - 10423.7085869859*cos(0.75*x) + 10080.7931168054*cos(1.0*x) - 1941.53683588888*cos(1.25*x) - 8417.57009120421*cos(1.5*x) + 13870.0260060319*cos(1.75*x) - 10589.8815345007*cos(2.0*x) + 172.419140031617*cos(2.25*x) + 12003.6310257316*cos(2.5*x) - 20459.5775693331*cos(2.75*x) + 22588.8053776129*cos(3.0*x) - 19250.3070298191*cos(3.25*x) + 13293.2509217136*cos(3.5*x) - 7521.57371616911*cos(3.75*x) + 3451.82526637215*cos(4.0*x) - 1244.91239685382*cos(4.25*x) + 330.327195096185*cos(4.5*x) - 55.5055593757824*cos(4.75*x) + 3.71888561074001*cos(5.0*x) + 25411.7788254111*cos(0.15*y) + 52544.2219993096*cos(0.3*y) - 9792.55870624065*cos(0.45*y) - 53496.0804232167*cos(0.6*y) + 9025.15827797479*cos(0.75*y) + 54784.4617595346*cos(0.9*y) - 13287.249209991*cos(1.05*y) - 55819.449675962*cos(1.2*y) + 23473.2956825209*cos(1.35*y) + 54151.5263866946*cos(1.5*y) - 42019.6287487197*cos(1.65*y) - 42244.3551605681*cos(1.8*y) + 68880.9238182208*cos(1.95*y) + 493.176188346702*cos(2.1*y) - 80191.987069957*cos(2.25*y) + 96585.3653178768*cos(2.4*y) - 62172.8082993734*cos(2.55*y) + 24473.0177668135*cos(2.7*y) - 5629.17407933511*cos(2.85*y) + 587.841752855767*cos(3.0*y) + 634.382402232455*cos(0.25*x - 0.15*y) + 636.407596037922*cos(0.25*x + 0.15*y) - 153.648618431164*cos(0.5*x - 0.3*y) - 155.638164383097*cos(0.5*x + 0.3*y) + 66.2283412967681*cos(0.75*x - 0.45*y) + 68.1596273749447*cos(0.75*x + 0.45*y) - 35.0680767039013*cos(1.0*x - 0.6*y) - 36.9198705282984*cos(1.0*x + 0.6*y) + 20.5189703481922*cos(1.25*x - 0.75*y) + 22.2726638379278*cos(1.25*x + 0.75*y) - 12.9581416096079*cos(1.5*x - 0.9*y) - 14.5974046647993*cos(1.5*x + 0.9*y) + 8.50345779316019*cos(1.75*x - 1.05*y) + 10.0151762006436*cos(1.75*x + 1.05*y) - 5.61611622219212*cos(2.0*x - 1.2*y) - 6.99058981696564*cos(2.0*x + 1.2*y) + 3.74817805852899*cos(2.25*x - 1.35*y) + 4.97888150417829*cos(2.25*x + 1.35*y) - 2.57272540323416*cos(2.5*x - 1.5*y) - 3.65690015876568*cos(2.5*x + 1.5*y) + 1.69544142485103*cos(2.75*x - 1.65*y) + 2.63380278589726*cos(2.75*x + 1.65*y) - 1.11273823082276*cos(3.0*x - 1.8*y) - 1.90865408395137*cos(3.0*x + 1.8*y) + 0.716915305206564*cos(3.25*x - 1.95*y) + 1.37745972667197*cos(3.25*x + 1.95*y) - 0.431352013281622*cos(3.5*x - 2.1*y) - 0.965309875593292*cos(3.5*x + 2.1*y) + 0.230896790291396*cos(3.75*x - 2.25*y) + 0.6494231768098*cos(3.75*x + 2.25*y) - 0.132586795046948*cos(4.0*x - 2.4*y) - 0.448660900254783*cos(4.0*x + 2.4*y) + 0.0850889575596586*cos(4.25*x - 2.55*y) + 0.311894204153997*cos(4.25*x + 2.55*y) - 0.0316608837401495*cos(4.5*x - 2.7*y) - 0.183768066878881*cos(4.5*x + 2.7*y) + 0.0108551085993822*cos(4.75*x - 2.85*y) + 0.101627612726815*cos(4.75*x + 2.85*y) - 0.00160141840788963*cos(5.0*x - 3.0*y) - 0.0429334528793861*cos(5.0*x + 3.0*y) - 51243.3578917042
f2 = -3.5
f1 = fourier_square + f2
```


```python
def newton3d(xn,yn,f1,f2):
    X = np.array([[xn],[yn]])
    f = np.array([[f1(xn,yn)],[f2(xn,yn)]])
    J = np.array([[dx_f1(xn,yn), dy_f1(xn,yn)],[dx_f2(xn),dy_f2(yn)]])
    J = np.linalg.inv(J)
    return X - np.dot(J,f)


dx_f1 = diff(f1, x)
dy_f1 = diff(f1, y)

f1 = lambdify([x,y], f1)
dx_f1 = lambdify([x,y], dx_f1)
dy_f1 = lambdify([x,y], dy_f1)

fwd = list(np.linspace(9999,99999,200))
rev = fwd[::-1]
tol = 0.000001 # threshold for convergence (difference between iterations must be below this value)
converge_dict = {'nx':[],'ny':[]}
for k,l in zip(rev, fwd):
    for f2 in [k*x-l*y,k*x+l*y]:
        dx_f2 = diff(f2, x)
        dy_f2 = diff(f2, y)
        
        f2 = lambdify([x,y], f2)
        dx_f2 = lambdify(x, dx_f2)
        dy_f2 = lambdify(y, dy_f2)
        
        # i = 0; n_x=0.50; n_y=0.501; prev_x=-9999; prev_y=-9999
        i = 0; n_x=round(random.uniform(-1, 1), 3); n_y=round(random.uniform(-1, 1), 3); prev_x=-9999; prev_y=-9999
        round(random.uniform(-1, 1), 3)
        while i < 10:
            n = newton3d(n_x, n_y, f1, f2)
            n_x=n[0][0]; n_y=n[1][0]
            if i == 9:
                if (abs(n_x - prev_x) < tol) and (abs(n_y - prev_y) < tol):
                    converge = 'Y'
                    converge_dict['nx'].append(n_x)
                    converge_dict['ny'].append(n_y)
                else:
                    converge = 'N'
            i += 1
            prev_x=n_x; prev_y=n_y

```


```python
converge_df = pd.DataFrame(converge_dict)
```


```python
ax1 = converge_df.plot.scatter(x='nx',y='ny')
ax1.set_xlim(-10,10)
ax1.set_ylim(-10,10)
```







    
![output_35_1](https://user-images.githubusercontent.com/36255172/145623099-0ba5c738-a138-41ff-aa5f-013f918c66a2.png)
    


And that's it! In this notebook we have successfully fit 3D data with a Fourier cosine series, and then used Newton's method to solve for specific points on the resulting surfaces. These concepts can be extended to higher dimensions. I'll post my code for a 4 dimensional version soon where I run it on the Iris dataset (a popular 4 feature dataset).
