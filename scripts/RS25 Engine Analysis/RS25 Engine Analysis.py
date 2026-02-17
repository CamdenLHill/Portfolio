# Camden Hill
# RS25 Engine Analysis

from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def Pre_Throat(rc_ratio,Conv_Length,Bx1c,Bx2c,TC_Factor,VC_Factor,CL,Thick_Function,t_samples):
    P0,P1,P2,P3 = [[-Conv_Length,rc_ratio],[-Conv_Length*Bx1c,rc_ratio],[-Conv_Length*Bx2c,1],[0,1]]
    Bx = (((1-t)**3)*P0[0]) + ((3*((1-t)**2)*t)*P1[0]) + (3*(1-t)*(t**2)*P2[0]) + ((t**3)*P3[0])
    By = (((1-t)**3)*P0[1]) + ((3*((1-t)**2)*t)*P1[1]) + (3*(1-t)*(t**2)*P2[1]) + ((t**3)*P3[1])
    tvals = np.linspace(0,1,t_samples)
    Bxvals = [Bx.subs(t,j) for j in tvals] # x values are not equally spaced across the range
    Byvals = [By.subs(t,j) for j in tvals]
    Vval = float(integrate(pi*((By**2)*diff(Bx,t)),(t,0,1)))
    Cyl_Volume = (CL*(float(pi*(TC_Factor**2)))) - (Vval*VC_Factor)
    Cyl_Length = (Cyl_Volume / (float(pi*((rc_ratio*TC_Factor)**2)))) / TC_Factor
    if Cyl_Volume <= 0:
        print("The Cylindrical section's volume is negative. This means the converging section is too long.\n The converging section is at least {} too long".format(Cyl_Length)) 
    Wall_xvals = list(np.linspace(-(Cyl_Length+Conv_Length),-Conv_Length,200))[:-1] + Bxvals[:-1]
    Wall_yvals = [rc_ratio for j in range(199)] + Byvals[:-1]
    Thick_yvals = [float(Wall_yvals[j]+Thick_Function.subs(x,Wall_xvals[j])) for j in range(len(Wall_xvals))]
    return Wall_xvals,Wall_yvals,Wall_xvals,Thick_yvals

def Post_Throat(re_ratio,Percent_Bell,Rd,Ell_percent,alpha_i,Bx1d,Thick_Function,t_samples):
    ellipse = (-(((Rd**2)*(1-((x/(Rd*Ell_percent))**2)))**0.5)) + (1 + Rd) # Generic divergence circle as defined by RAO. Empirical & just used to determine 15 degree conic
    x_ints1 = float(solve(diff(ellipse,x)-tan(alpha_i),x)[0])
    y_ints1 = ellipse.subs(x,x_ints1)
    Ex_vals = list(np.linspace(0,x_ints1,100))
    Ey_vals = [ellipse.subs(x,j) for j in Ex_vals]
    cone_Length = float((re_ratio-1)/tan(pi*(15/180)))
    P0,P1,P2 = [x_ints1,y_ints1],[Bx1d*Percent_Bell*(cone_Length-y_ints1),(tan(alpha_i)*((Bx1d*Percent_Bell*(cone_Length-y_ints1))-x_ints1))+y_ints1],[cone_Length*Percent_Bell,re_ratio]
    tvals = np.linspace(0,1,t_samples)
    Bx = (((1-t)**2)*P0[0]) + ((2*(1-t)*t)*P1[0]) + ((t**2)*P2[0])
    By = (((1-t)**2)*P0[1]) + ((2*(1-t)*t)*P1[1]) + ((t**2)*P2[1])
    Wall_xvals = Ex_vals[:-1] + [Bx.subs(t,i) for i in tvals]
    Wall_yvals = Ey_vals[:-1] + [By.subs(t,i) for i in tvals]
    Thick_yvals = [float(Wall_yvals[j]+Thick_Function.subs(x,Wall_xvals[j])) for j in range(len(Wall_xvals))]
    alpha_e = float(atan((diff(By,t)/diff(Bx,t)).subs(t,1)) * 180 / pi)
    return Wall_xvals,Wall_yvals,Wall_xvals,Thick_yvals,alpha_e

def NewtonRaphson(f,fprime,a,guess,error):
   i = 0
   while True:
     i += 1
     guess -= N(f.subs(a,guess)) / N(fprime.subs(a,guess))
     if np.absolute(N(f.subs(a,guess))) < error:
       return guess,i
     if i >= 250:
       return False,guess,i

def Area_to_Mach(A,guess, gamma=1.4):
   M = symbols("M")
   f = (1/M**2*(2/(gamma+1)*(1+(gamma-1)/2*M**2))**((gamma+1)/(gamma-1)))**.5 - A
   fprime = diff(f,M)
   return abs(NewtonRaphson(f,fprime,M,guess,10**-9)[0])

def Mach_to_Area(M, gamma=1.4):
   return (1/M**2*(2/(gamma+1)*(1+(gamma-1)/2*M**2))**((gamma+1)/(gamma-1)))**.5

##### Unit Conversions
ft_m = 0.3048
lbs_kg = 0.45359237
lbf_N = 4.4482216153
psi_Pa = 6894.7572931783
##### Generic Info
g_o = 9.80665
Length = (168 / 12) * ft_m
Width = (96 / 12) * ft_m
Ae_ratio = 69
Ae = float(pi*((Width/2)**2))
At = Ae / Ae_ratio
nu = 1 # 0.975 # Overall efficiency. Includes improper combustion, viscous effects, nozzle divergence inefficiency
##### All Variables given at full Power Level. Meaning maximum mass flow???
f = 6.03 # LOX/H by mass
Po = 2994 * psi_Pa # * 6.095
Me = symbols('y R Me')
R,To = [461.52,3588.15]
ys = [1.05,1.1,1.3,1.5,1.26] # [1.26]
##### Vacuum / Power Level 109%
Isp_v = 452.3
T_v = 512300 * lbf_N
m_v = (T_v/nu) / (Isp_v*g_o) # kg/s # checks out with the fuel tanks on for the space shuttle
print("Vacuum Thrust (N) = {}, Vacuum mass flow rate = {}".format(T_v,m_v))
##### Sea Level / Power Level 109%
Isp_s = 366
T_s = 418000 * lbf_N
m_s = (T_s/nu) / (Isp_s*g_o) # kg/s # checks out with the fuel tanks on for the space shuttle
print("Sea Level Thrust (N) = {}, Sea Level mass flow rate = {}".format(T_s,m_s))
#############################
# To_v = (sqrt(y/R)*((2/(y+1))**((y+1)/(2*(y-1))))*(Po/m_v)*At)**2 # St. Venant's Equation to find To
for i in ys:
   y = i
   Me = Area_to_Mach(Ae_ratio,3,y)
   To_T_v = (1+(((y-1)/2)*(Me**2)))
   Te_v = To / To_T_v
   ve_v = float(Me*sqrt(y*R*Te_v))
   Pe_v = Po / (To_T_v**(y/(y-1)))
   Tm_v = (m_v*ve_v)
   Tp_v = (Pe_v*Ae)
   Ts_v = Tm_v + Tp_v
   print("Exit Area = {}, Mass Flow Rate = {}, Chamber Pressure (atm) = {}, Chamber Temperature = {}, Exit Velocity = {}, Exit Pressure (atm) = {}".format(Ae,m_v,Po/101325,To,ve_v,Pe_v/101325))
   print("Calculated Momentum thrust = {}, Calculated Pressure Thrust = {}, Calculated Thrust = {}, Actual Thrust = {}\n".format(Tm_v,Tp_v,Ts_v,T_v))

t,x = symbols("t x")
colors = ["blue","red","orange","green","pink","gray","purple","yellow","brown","cyan","black","blue","red","orange","green","pink","gray","purple","yellow","brown","cyan","black"]

Ac_ratio = 3.5 #2.5-3.5 generally
### Post-Throat Settings
re_ratio = Ae_ratio**0.5
Percent_Bell = 0.8 #0.8
Rd = 0.6 # 0.382
Ell_percent = 0.8 # at High Ae/At barely affects exit angle. At low it plays a larger role
alpha_i = float(pi*(33/180)) # At High Ae/At each degree increase decreases exit angle by roughly 1 degree. At low each degree is worth about 0.4 exit angle
Bx1d = 0.42 # At High Ae/At reduces angle by half degrees for each 0.01 increase. But at lower Ae/At reduces by quarters

### Pre-Throat Settings
rc_ratio = Ac_ratio**0.5
Bx1c = 0.575
Bx2c = 0.2
Conv_Length = 3.5

### Thickness Settings
# The Thickness Function is in terms of the standard basis. Can be used later on with a spline and iterative design to get a "Perfect" Thickness
t_samples = 200
# Thick_Function = 0.1 + (abs(0.35*cos(x))) + (0.1*((1/(0.25*((float(2*pi))**0.5)))*(E**(-0.5*((x*4)**2))))) # use Float(#) if f=const. & not f(x)=eq
Thick_Function = 0.1 + (0.4*((1/(0.25*((float(2*pi))**0.5)))*(E**(-0.5*((x*4)**2)))))
# Thick_Function = Float(0.1)
TC_Factor = float(sqrt(At/pi)) # The conversion rate of the throat !!radius!! from a standard basis to real world dimensions. so put in the !!radius!! in real world dimensions
VC_Factor = TC_Factor**3 # Converts the standardize volume to a real volume. Used for L* and Cylinder Length.
CL = 2.5 #### !!!Characteristic Length!!! #### in m. for

Wallc_xvals,Wallc_yvals,Thickc_xvals,Thickc_yvals = Pre_Throat(rc_ratio,Conv_Length,Bx1c,Bx2c,TC_Factor,VC_Factor,CL,Thick_Function,t_samples)
Walld_xvals,Walld_yvals,Thickd_xvals,Thickd_yvals,alpha_e = Post_Throat(re_ratio,Percent_Bell,Rd,Ell_percent,alpha_i,Bx1d,Thick_Function,t_samples)

###################### Plot of Nozzle Geometry
plt.plot(Wallc_xvals,Wallc_yvals,colors[0])
plt.plot(Walld_xvals,Walld_yvals,colors[1])
# plt.plot(Thickc_xvals,Thickc_yvals,colors[2])
# # plt.plot(Thickd_xvals,Thickd_yvals,colors[3])
plt.plot(Wallc_xvals,[-i for i in Wallc_yvals],colors[0])
plt.plot(Walld_xvals,[-i for i in Walld_yvals],colors[1])
# plt.plot(Thickc_xvals,[-i for i in Thickc_yvals],colors[2])
# plt.plot(Thickd_xvals,[-i for i in Thickd_yvals],colors[3])
plt.xlabel("Non-Dimensionalized Length (x)")
plt.ylabel("Non-Dimensionalized Radius (r)")
plt.title("Graph of Non-Dimensionalized Bell Nozzle with an Area Ratio of 69")
plt.show()
print()

###################### Plot of thermodynamic properties across the length of the nozzle
rhoo = Po/(R*To)
Machsc,Tvalsc,Pvalsc,rhovalsc,vvalsc,Machsd,Tvalsd,Pvalsd,rhovalsd,vvalsd = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(Wallc_xvals)):
   Mval = Area_to_Mach((Wallc_yvals[i]**2),.4,y)
   To_T = (1+(((y-1)/2)*(Mval**2)))
   T = To / To_T
   Machsc.append(Mval)
   Tvalsc.append(T)
   Pvalsc.append(Po / (To_T**(1/(y-1))))
   rhovalsc.append(rhoo / (To_T**((y)/(y-1))))
   vvalsc.append(float(Mval*sqrt(y*R*T)))
for i in range(len(Walld_xvals)):
   Mval = Area_to_Mach((Walld_yvals[i]**2),3,y)
   To_T = (1+(((y-1)/2)*(Mval**2)))
   T = To / To_T
   Machsd.append(Mval)
   Tvalsd.append(T)
   Pvalsd.append(Po / (To_T**(1/(y-1))))
   rhovalsd.append(rhoo / (To_T**((y)/(y-1))))
   vvalsd.append(float(Mval*sqrt(y*R*T)))
combx,comby = [Wallc_xvals + Walld_xvals,Wallc_yvals + Walld_yvals]
Machs,Tvals,Pvals,rhovals,vvals = [Machsc+Machsd,Tvalsc+Tvalsd,Pvalsc+Pvalsd,rhovalsc+rhovalsd,vvalsc+vvalsd]
Machs,Tvals,Pvals,rhovals,vvals = [[i for i in np.array(Machs)/max(Machs)],[i for i in np.array(Tvals)/max(Tvals)],[i for i in np.array(Pvals)/max(Pvals)],[i for i in np.array(rhovals)/max(rhovals)],[i for i in np.array(vvals)/max(vvals)]]
plt.plot(combx,Machs,color="pink",label="Mach")
plt.plot(combx,Tvals,color="red",label="Temperature")
plt.plot(combx,Pvals,color="blue",label="Pressure")
plt.plot(combx,rhovals,color="green",label="Density")
plt.plot(combx,vvals,color="orange",label="Velocity")
plt.plot([0 for i in range(100)],np.linspace(0,1,100),color="black",label="Mach = 1")
plt.xlabel("Non-Dimensionalized Length (x)")
plt.ylabel("Scaled Thermodynamic Properties")
plt.title("Non-Dimensionalized Graph of Thermodynamic Properties through a Supersonic Bell Nozzle")
plt.legend()
plt.show()
print()

###################### Plot of thermodynamic properties as a function of Mach Number
rhoo,Machs = [Po/(R*To),np.linspace(0.01,10,250)]
Tvals,Pvals,rhovals,vvals = [[],[],[],[]]
for i in Machs:
   To_T = (1+(((y-1)/2)*(i**2)))
   T = To / To_T
   Tvals.append(T)
   Pvals.append(Po / (To_T**(1/(y-1))))
   rhovals.append(rhoo / (To_T**((y)/(y-1))))
   vvals.append(float(i*sqrt(y*R*T)))
Tvals,Pvals,rhovals,vvals = [[i for i in np.array(Tvals)/max(Tvals)],[i for i in np.array(Pvals)/max(Pvals)],[i for i in np.array(rhovals)/max(rhovals)],[i for i in np.array(vvals)/max(vvals)]]
plt.plot(Machs,Tvals,color="red",label="Temperature")
plt.plot(Machs,Pvals,color="blue",label="Pressure")
plt.plot(Machs,rhovals,color="green",label="Density")
plt.plot(Machs,vvals,color="orange",label="Velocity")
plt.plot([1 for i in range(100)],np.linspace(0,1,100),color="black",label="Mach = 1")
plt.xlabel("Mach Number")
plt.ylabel("Scaled Thermodynamic Properties")
plt.title("Thermodynamic Properties vs. Mach Number through a Supersonic Bell Nozzle")
plt.legend()
plt.show()
print()

###################### Plot of thrust over working altitude
pa_s = np.linspace(1,0,100)
plt.plot([1,0],[T_s,T_v],color="orange",label="Linear Interpolation of Reported Thrust")
plt.plot(pa_s,[(m_v*ve_v)+((Pe_v-i)*Ae) for i in pa_s*101325],color="red",label="Thrust Equation with Vacuum Values")
plt.xlabel("Back Pressure/Atmospheric Pressure (atm)")
plt.ylabel("Total Thrust (N)")
plt.title("Total Thrust vs. Back Pressure")
plt.ylim(0,(T_v*1.01))
plt.legend()
plt.show()
print()