from sympy import sin, cos, symbols, lambdify, diff, exp
import numpy as np
import random

x, y = symbols('x y')
f1 = 5453.97363194277*cos(0.25*x) + 3096.4157373821*cos(0.5*x) - 10423.7085869859*cos(0.75*x) + 10080.7931168054*cos(1.0*x) - 1941.53683588888*cos(1.25*x) - 8417.57009120421*cos(1.5*x) + 13870.0260060319*cos(1.75*x) - 10589.8815345007*cos(2.0*x) + 172.419140031617*cos(2.25*x) + 12003.6310257316*cos(2.5*x) - 20459.5775693331*cos(2.75*x) + 22588.8053776129*cos(3.0*x) - 19250.3070298191*cos(3.25*x) + 13293.2509217136*cos(3.5*x) - 7521.57371616911*cos(3.75*x) + 3451.82526637215*cos(4.0*x) - 1244.91239685382*cos(4.25*x) + 330.327195096185*cos(4.5*x) - 55.5055593757824*cos(4.75*x) + 3.71888561074001*cos(5.0*x) + 25411.7788254111*cos(0.15*y) + 52544.2219993096*cos(0.3*y) - 9792.55870624065*cos(0.45*y) - 53496.0804232167*cos(0.6*y) + 9025.15827797479*cos(0.75*y) + 54784.4617595346*cos(0.9*y) - 13287.249209991*cos(1.05*y) - 55819.449675962*cos(1.2*y) + 23473.2956825209*cos(1.35*y) + 54151.5263866946*cos(1.5*y) - 42019.6287487197*cos(1.65*y) - 42244.3551605681*cos(1.8*y) + 68880.9238182208*cos(1.95*y) + 493.176188346702*cos(2.1*y) - 80191.987069957*cos(2.25*y) + 96585.3653178768*cos(2.4*y) - 62172.8082993734*cos(2.55*y) + 24473.0177668135*cos(2.7*y) - 5629.17407933511*cos(2.85*y) + 587.841752855767*cos(3.0*y) + 634.382402232455*cos(0.25*x - 0.15*y) + 636.407596037922*cos(0.25*x + 0.15*y) - 153.648618431164*cos(0.5*x - 0.3*y) - 155.638164383097*cos(0.5*x + 0.3*y) + 66.2283412967681*cos(0.75*x - 0.45*y) + 68.1596273749447*cos(0.75*x + 0.45*y) - 35.0680767039013*cos(1.0*x - 0.6*y) - 36.9198705282984*cos(1.0*x + 0.6*y) + 20.5189703481922*cos(1.25*x - 0.75*y) + 22.2726638379278*cos(1.25*x + 0.75*y) - 12.9581416096079*cos(1.5*x - 0.9*y) - 14.5974046647993*cos(1.5*x + 0.9*y) + 8.50345779316019*cos(1.75*x - 1.05*y) + 10.0151762006436*cos(1.75*x + 1.05*y) - 5.61611622219212*cos(2.0*x - 1.2*y) - 6.99058981696564*cos(2.0*x + 1.2*y) + 3.74817805852899*cos(2.25*x - 1.35*y) + 4.97888150417829*cos(2.25*x + 1.35*y) - 2.57272540323416*cos(2.5*x - 1.5*y) - 3.65690015876568*cos(2.5*x + 1.5*y) + 1.69544142485103*cos(2.75*x - 1.65*y) + 2.63380278589726*cos(2.75*x + 1.65*y) - 1.11273823082276*cos(3.0*x - 1.8*y) - 1.90865408395137*cos(3.0*x + 1.8*y) + 0.716915305206564*cos(3.25*x - 1.95*y) + 1.37745972667197*cos(3.25*x + 1.95*y) - 0.431352013281622*cos(3.5*x - 2.1*y) - 0.965309875593292*cos(3.5*x + 2.1*y) + 0.230896790291396*cos(3.75*x - 2.25*y) + 0.6494231768098*cos(3.75*x + 2.25*y) - 0.132586795046948*cos(4.0*x - 2.4*y) - 0.448660900254783*cos(4.0*x + 2.4*y) + 0.0850889575596586*cos(4.25*x - 2.55*y) + 0.311894204153997*cos(4.25*x + 2.55*y) - 0.0316608837401495*cos(4.5*x - 2.7*y) - 0.183768066878881*cos(4.5*x + 2.7*y) + 0.0108551085993822*cos(4.75*x - 2.85*y) + 0.101627612726815*cos(4.75*x + 2.85*y) - 0.00160141840788963*cos(5.0*x - 3.0*y) - 0.0429334528793861*cos(5.0*x + 3.0*y) - 51243.3578917042 - 3.5


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

fwd = list(np.linspace(9999,99999,50))
rev = fwd[::-1]
tol = 0.000001
converge_dict = {'nx':[],'ny':[]}
for k,l in zip(rev, fwd):
    for f2 in [k*x-l*y,k*x+l*y]:
        # f2 = k*x - l*y
        #print(f'{k}*x - {l}*y')
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
            # print(abs(n_x - prev_x), abs(n_y - prev_y))
            if i == 9:
                if (abs(n_x - prev_x) < tol) and (abs(n_y - prev_y) < tol):
                    converge = 'Y'
                    print(n_x,n_y,converge)
                    converge_dict['nx'].append(n_x)
                    converge_dict['ny'].append(n_y)
                else:
                    converge = 'N'
                # print(n_x,n_y,converge)
            i += 1
            prev_x=n_x; prev_y=n_y
