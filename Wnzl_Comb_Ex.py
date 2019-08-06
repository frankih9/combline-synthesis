from Wnzl_Funcs import*
import numpy as np

#IV. Combline Filter Synthesis Example                                               
#1)
f1 = 1.2e9
f2 = 1.8e9
ripple = 0.1 #dB
#2)
AH = 30     #High side attenuation in dB
fH = 2.4e9  #High side attenuation frequency   
AL = 20     #Los side attenuation in dB
fL = 0.5e9  #Low side attenuation frequency
#3)
fS = 7.418e9 #Quarterwave frequency or stop band frequency
#In the paper, fS = 7.148GHz is incorrect

#Equations for combline filters from Table I.
N_Order, omega1, omega2, e, fo = Comb_Req_Order(f1, f2, ripple, AL, fL, AH, fH, fS)

#N_Order = 4
#EQN(5): Form the polynomial E+ZF   
EpZF = Comb_EpZF(N_Order, omega1, omega2)

#EQN(6): Extract polynomial E and form the polynomial E+ZF/sqr(1+e^2)
E, EpZF_rf = EpZF_rf(EpZF, e)    

#EQN(6): Find the factored quadratics
EpZF_rf_fact = EpZF_rf_Poly2(EpZF_rf)
        
#EQN(7): Generate coefficients p, q, r(lower case gamma from paper)
p, q, r = p_q_r_Array(EpZF_rf_fact, omega1, omega2)   

#EQN(8): Extract A and B from A+Bsqrt((Z^2-1)(omega^2-(omega1^2)(Z^2)))      
sorted_combo = Sorted_Combo(N_Order)
A, B = ApBsqrt_fact(N_Order, e, p, q, r, omega1, omega2, sorted_combo)

#EQN(9):Form polynomial Y(Z^2)'s numerator Yn=A+eE 
YZn, YZd = YofZ2(A, E, B, e, omega1, omega2)

#Convert Y(Z^2) to Y(S)/S by using EQN(2)
YSn, YSd = YofS(YZn, YZd, omega1, omega2)

#Extract capacitors and inductors for equal capacitor values
cap_array, ind_array = Comb_Element_Extract(YSn, YSd, N_Order)

#Simulation frequency range in Hz
f = np.arange(20e6, 4e9, 20e6)

#Applying Richard's Transform
w = np.tan(np.pi/2*f/fo)

#Simulate using ABCD 
S11_dB, S21_dB = Eval_Elements(cap_array, ind_array, w)

#Normalize frequency to GHz and plot
fplot = f/1e9
Plot_S(S11_dB, S21_dB, fplot, fH/1e9, AH, 'Combline Example')

print('Sections:    ',N_Order)
print('Shunt Caps:  ',cap_array[::2])
print('Shunt Inds:  ',ind_array[::2])
print('Series Inds: ',ind_array[1::2])

Lsho = np.array([])
sh0 = 1/(1/ind_array[0]+1/ind_array[1])
Lsho = np.append(Lsho, sh0)
for i in np.arange(2, len(ind_array)-2, 2):
    sh = 1/(1/ind_array[i]+1/ind_array[i-1]+1/ind_array[i+1])
    Lsho = np.append(Lsho, sh)
shn = 1/(1/ind_array[-1]+1/ind_array[-2])    
Lsho = np.append(Lsho, shn)

print('Alone Inds:  ',Lsho)