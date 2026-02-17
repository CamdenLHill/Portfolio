# Camden Hill

import numpy as np
# import matplotlib.pyplot as plt
from sympy import *

def COD_rot_stat(T01,P01,Trise,DegreeR,polyn,Ca,Cp,gamma,R,massa_flow,rt_i,N):
    ##### Rotor Entrance Calc
    Tinc_rotor = T01
    Prat_rotor = (Tinc_rotor/T01)**(polyn/(polyn-1))
    Pinc_rotor = Prat_rotor * P01
    Tstat_inc = Tinc_rotor - ((Ca**2)/(2*Cp))
    Pstat_inc = Pinc_rotor * ((Tstat_inc/Tinc_rotor)**(gamma/(gamma-1)))
    Rhorstat_inc = Pstat_inc/(R*Tstat_inc)
    Arotor_inc = massa_flow / (Rhorstat_inc*Ca)
    rhrotor_inc = ((rt_i**2-(Arotor_inc/np.pi)))**0.5
    ##### Stator Entrance Calc
    Tinc_stator = T01 + (Trise * DegreeR)
    Prat_stator = (Tinc_stator/T01)**(polyn/(polyn-1))
    Pinc_stator = Prat_stator * P01
    Tstat_inc = Tinc_stator - ((Ca**2)/(2*Cp))
    Pstat_inc = Pinc_stator * ((Tstat_inc/Tinc_stator)**(gamma/(gamma-1)))
    Rhosstat_inc = Pstat_inc/(R*Tstat_inc)
    Astator_inc = massa_flow / (Rhosstat_inc*Ca)
    rhstator_inc = ((rt_i**2-(Astator_inc/np.pi)))**0.5
    ##### Change mean diameter and rotational speed
    rrotor_mean = (rt_i + rhrotor_inc)/2
    rstator_mean = (rt_i + rhstator_inc)/2
    U1 = rrotor_mean * 2 * np.pi * N
    U2 = rstator_mean * 2 * np.pi * N
    return U1,Tinc_rotor,Pinc_rotor,Rhorstat_inc,Arotor_inc,rhrotor_inc,U2,Tinc_stator,Pinc_stator,Rhosstat_inc,Astator_inc,rhstator_inc

#### Project Example
Comp_ratio = 4.15 
massa_flow = 20 # kg/s
T03 = 1100 # Kelvin

Takeoff_thrust = 12000 #Newtons
P01 = 1.01 * (10**5) # Pascals
T01 = 288 # Kelvin

Ut = 350 # m/s
Ca = 150 # m/s # 150 - 200 m/s
hub_tip = 0.5 # 0.4 - 0.6

# P01 = 102000
# T01 = 335
mflow = 5
gamma = 1.4
Cp = 1005
R = 287.16
print("Assignment Example:\n")
iterations = 2
P1,T1 = [P01,T01]
for i in range(iterations): # This specific application doesn't require iteration but its here regardless.
    rho = P1/(R*T1)
    rt_i = (massa_flow / (np.pi*rho*Ca*(1-(hub_tip**2)))) ** 0.5
    rh_i = (rt_i*hub_tip)
    A1 = np.pi*((rt_i**2)-(rh_i**2))
    # Ca = mflow/(rho*A1)
    T1 = T01 - ((Ca**2)/(2*Cp))
    P1 = P01 * ((T1/T01)**(gamma/(gamma-1)))
    # print(T1)
    # print(rho)
    # print(A1)
print("T1 = {} K".format(T1))
print("rho1 = {} kg/m^3".format(rho))
print("Inlet hub radius = {} m".format(rh_i))
print("Inlet tip radius = {} m".format(rt_i))
print("Area = {} m^2".format(A1))

N = Ut / (2*np.pi*rt_i)
print("N = {} rev/s".format(N))

###### Arbitrary change in N
N = 250
print("N = {} rev/s".format(N))
Ut = N*2*np.pi*rt_i
print("Ut = {} m/s".format(Ut))
######

V1t = ((Ut**2)+(Ca**2)) ** 0.5
print("V1t = {} m/s".format(V1t))
a = (gamma*R*T1) ** 0.5
print("a = {} m/s".format(a))
M1t = V1t / a
print("M1t = {}".format(M1t))

r_mean = (rt_i + rh_i) / 2
P02 = Comp_ratio * P01
print("P02 = {} P01".format(P02))

polytropic_efficiency = 0.9
n = 1/(1 - (((gamma-1)/(polytropic_efficiency*gamma))))
print("(n-1)/n = {}".format((n-1)/n))
T02 = T01 * (Comp_ratio**((n-1)/n))
print("T02 = {} K".format(T02))

T2 = T02 - ((Ca**2)/(2*Cp))
P2 = P02*((T2/T02)**(gamma/(gamma-1)))
rho2 = P2/(R*T2)
print("T2 = {} K".format(T2))
print("P2 = {} P01".format(P2))
print("rho2 = {} kg/m^3".format(rho2))
A2 = massa_flow / (rho2 * Ca)
print("A2 = {} m^2".format(A2))

blade_height = A2 / (2*np.pi*r_mean)
print("blade height = {}".format(blade_height))
rt_o = r_mean + (blade_height/2)
rh_o = r_mean - (blade_height/2)
print("Outlet tip radius = {} m".format(rt_o))
print("Outlet hub radius = {} m".format(rh_o))

change_stagT = T02 - T01
print("Change in stagnation temperature = {} K".format(change_stagT))

######## Estimation of Stages
U = 2*np.pi*r_mean*N
print("U = {} m/s".format(U))
beta_1 = np.arctan(U / Ca)
print("Beta 1 = {} degrees".format((180/np.pi)*beta_1))
V1 = Ca / np.cos(beta_1)
print("V1 = {} m/s".format(V1))
#### De Haller Criterion
V2 = V1 * 0.72
beta_2 = np.arccos(Ca/V2)
print("Beta 2 = {} degrees".format((180/np.pi)*beta_2))
change_stagT_stage = (U*Ca*(np.tan(beta_1)-np.tan(beta_2)))/(Cp)
print("Approximate change in stagnation temperature per stage = {}".format(change_stagT_stage))
num_stages = change_stagT / change_stagT_stage
print("Approximate number of stages = {}".format(num_stages))

######## Stage-by-stage design ###################### Constant Diameter (Example)
U = 266.6
CD = {}
num_stages = 7
work_done_factors = [0.98,0.93,0.88,0.83,0.83,0.83,0.83]
degrees_of_reactions = [0,0.7,0.5,0.5,0.5,0.5,0.5]
dtempi,dtempf = [20,20]
# temperature_rises = [dtempi] + [(change_stagT-(dtempi+dtempf))/(num_stages-2) for i in range(num_stages-2)] + [dtempf]
temperature_rises = [dtempi] + [25] + [24 for i in range(num_stages-3)]
temperature_rises += [change_stagT-sum(temperature_rises)]
rhubs = np.linspace(rh_i,((rt_i**2-(A2/np.pi)))**0.5,num_stages*2)
rmeans = [(rt_i+i)/2 for i in rhubs]
print("Stage temperature rises = {}".format(temperature_rises))
for i in range(num_stages):
    CD[i+1] = {"WD":work_done_factors[i],"DR":degrees_of_reactions[i],"TR":temperature_rises[i]}
for i in range(1,num_stages+1):
    print("\nStage {}:".format(i))
    if i == 1: # if the first stage
        dCw = (Cp*CD[i]["TR"]) / (CD[i]["WD"]*U)
        Cw1,Cw2 = [0,dCw]
        beta_1,beta_2,alpha_1,alpha_2 = [(180/np.pi)*np.arctan(U/Ca),(180/np.pi)*np.arctan((U-Cw2)/Ca),(180/np.pi)*np.arctan(Cw1/Ca),(180/np.pi)*np.arctan(Cw2/Ca)]
        print("Cw1 = {} m/s\nCw2 = {} m/s\nbeta 1 = {} deg.\nbeta 2 = {} deg.\nalpha 1 = {} deg.\nalpha 2 = {} deg.\n".format(Cw1,Cw2,beta_1,beta_2,alpha_1,alpha_2))
        CD[i]["Cw1"],CD[i]["Cw2"],CD[i]["beta1"],CD[i]["beta2"],CD[i]["alpha1"],CD[i]["alpha2"] = [Cw1,Cw2,beta_1,beta_2,alpha_1,alpha_2]
        deHaller = np.cos((np.pi/180)*beta_1)/np.cos((np.pi/180)*beta_2)
        print("de Haller number = {}".format(deHaller))
        CD[i]["deHaller"] = deHaller
        pressure_ratio = (1 + (polytropic_efficiency*CD[i]["TR"]/T01)) ** (gamma/(gamma-1))
        P0o,T0o = [P01 * pressure_ratio,T01 + CD[i]["TR"]]
        print("pressure ratio = {}\nOutlet pressure = {} Pa\nOutlet Temperature = {} K".format(pressure_ratio,P0o,T0o))
        CD[i]["PR"],CD[i]["P0o"],CD[i]["T0o"] = [pressure_ratio,P0o,T0o]
        CD[i]["DR"] = 1 - ((Cw2+Cw1)/(2*U))
        print("degree of reaction = {}".format(CD[i]["DR"]))
    elif i == num_stages: # last stage
        pressure_ratio = (Comp_ratio*P01) / CD[i-1]["P0o"]
        CD[i]["TR"] = ((pressure_ratio**((gamma-1)/gamma))-1)*CD[i-1]["T0o"]/polytropic_efficiency
        ######
        A = np.array([[1,-1],[1,1]])
        B = np.array([CD[i]["TR"]/((CD[i]["WD"]*U*Ca)/Cp),CD[i]["DR"]/((Ca)/(2*U))])
        x = np.linalg.solve(A,B)
        beta_1,beta_2 = [(180/np.pi)*np.arctan(x[0]),(180/np.pi)*np.arctan(x[1])]
        alpha_1,alpha_2 = [(180/np.pi)*np.arctan((U/Ca)-np.tan((np.pi/180)*beta_1)),(180/np.pi)*np.arctan((U/Ca)-np.tan((np.pi/180)*beta_2))]
        Cw1,Cw2 = [Ca*np.tan((np.pi/180)*alpha_1),Ca*np.tan((np.pi/180)*alpha_2)]
        print("Cw1 = {} m/s\nCw2 = {} m/s\nbeta 1 = {} deg.\nbeta 2 = {} deg.\nalpha 1 = {} deg.\nalpha 2 = {} deg.\n".format(Cw1,Cw2,beta_1,beta_2,alpha_1,alpha_2))
        CD[i]["Cw1"],CD[i]["Cw2"],CD[i]["beta1"],CD[i]["beta2"],CD[i]["alpha1"],CD[i]["alpha2"] = [Cw1,Cw2,beta_1,beta_2,alpha_1,alpha_2]
        CD[i-1]["alpha3"] = alpha_1
        RdeHaller = np.cos((np.pi/180)*beta_1)/np.cos((np.pi/180)*beta_2)
        print("Rotor de Haller number = {}".format(RdeHaller))
        SdeHaller = np.cos((np.pi/180)*alpha_2)/np.cos((np.pi/180)*alpha_1) # np.cos((np.pi/180)*alpha_2)/np.cos((np.pi/180)*alpha_1)
        print("Stator de Haller number = {}".format(SdeHaller))
        CD[i]["RdeHaller"],CD[i]["SdeHaller"] = [RdeHaller,SdeHaller]
        pressure_ratio = (1 + (polytropic_efficiency*CD[i]["TR"]/CD[i-1]["T0o"])) ** (gamma/(gamma-1))
        P0o,T0o = [CD[i-1]["P0o"] * pressure_ratio,CD[i-1]["T0o"] + CD[i]["TR"]]
        print("pressure ratio = {}\nOutlet pressure = {} Pa\nOutlet Temperature = {} K".format(pressure_ratio,P0o,T0o))
        CD[i]["PR"],CD[i]["P0o"],CD[i]["T0o"] = [pressure_ratio,P0o,T0o]
        CD[i]["DR"] = 1 - ((Cw2+Cw1)/(2*U))
        print("degree of reaction = {}".format(CD[i]["DR"]))
    else: # middle stages
        A = np.array([[1,-1],[1,1]])
        B = np.array([CD[i]["TR"]/((CD[i]["WD"]*U*Ca)/Cp),CD[i]["DR"]/((Ca)/(2*U))])
        x = np.linalg.solve(A,B)
        beta_1,beta_2 = [(180/np.pi)*np.arctan(x[0]),(180/np.pi)*np.arctan(x[1])]
        alpha_1,alpha_2 = [(180/np.pi)*np.arctan((U/Ca)-np.tan((np.pi/180)*beta_1)),(180/np.pi)*np.arctan((U/Ca)-np.tan((np.pi/180)*beta_2))]
        Cw1,Cw2 = [Ca*np.tan((np.pi/180)*alpha_1),Ca*np.tan((np.pi/180)*alpha_2)]
        print("Cw1 = {} m/s\nCw2 = {} m/s\nbeta 1 = {} deg.\nbeta 2 = {} deg.\nalpha 1 = {} deg.\nalpha 2 = {} deg.\n".format(Cw1,Cw2,beta_1,beta_2,alpha_1,alpha_2))
        CD[i]["Cw1"],CD[i]["Cw2"],CD[i]["beta1"],CD[i]["beta2"],CD[i]["alpha1"],CD[i]["alpha2"] = [Cw1,Cw2,beta_1,beta_2,alpha_1,alpha_2]
        CD[i-1]["alpha3"] = alpha_1
        RdeHaller = np.cos((np.pi/180)*beta_1)/np.cos((np.pi/180)*beta_2)
        print("Rotor de Haller number = {}".format(RdeHaller))
        SdeHaller = np.cos((np.pi/180)*alpha_2)/np.cos((np.pi/180)*alpha_1) # np.cos((np.pi/180)*alpha_2)/np.cos((np.pi/180)*alpha_1)
        print("Stator de Haller number = {}".format(SdeHaller))
        CD[i]["RdeHaller"],CD[i]["SdeHaller"] = [RdeHaller,SdeHaller]
        pressure_ratio = (1 + (polytropic_efficiency*CD[i]["TR"]/CD[i-1]["T0o"])) ** (gamma/(gamma-1))
        P0o,T0o = [CD[i-1]["P0o"] * pressure_ratio,CD[i-1]["T0o"] + CD[i]["TR"]]
        print("pressure ratio = {}\nOutlet pressure = {} Pa\nOutlet Temperature = {} K".format(pressure_ratio,P0o,T0o))
        CD[i]["PR"],CD[i]["P0o"],CD[i]["T0o"] = [pressure_ratio,P0o,T0o]
        CD[i]["DR"] = 1 - ((Cw2+Cw1)/(2*U))
        print("Estimated degree of reaction = {}".format(CD[i]["DR"]))


#### Project Compressor
Comp_ratio = 5 
massa_flow = 8.1 # kg/s
T03 = 1100 # Kelvin

Takeoff_thrust = 12000 #Newtons
P01 = 101325 # Pascals
T01 = 288.15 # Kelvin

hub_tip = 0.4 # 0.4 - 0.6

gamma = 1.4
Cp = 1005
R = 287.16

polytropic_efficiency = 0.89
a_inlet = (gamma*R*T01)**0.5
Mach_inlet = 0.5
Ca = Mach_inlet * a_inlet
N = 25650 / 60 # rps
print("5-stage Compressor using Constant Outer Diameter:\n")
iterations = 2
P1,T1 = [P01,T01]
for i in range(iterations): # This specific application doesn't require iteration but its here regardless.
    rho = P1/(R*T1)
    rt_i = (massa_flow / (np.pi*rho*Ca*(1-(hub_tip**2)))) ** 0.5
    rh_i = (rt_i*hub_tip)
    A1 = np.pi*((rt_i**2)-(rh_i**2))
    # Ca = mflow/(rho*A1)
    T1 = T01 - ((Ca**2)/(2*Cp))
    P1 = P01 * ((T1/T01)**(gamma/(gamma-1)))
    # print(T1)
    # print(rho)
    # print(A1)
print("T1 = {} K".format(T1))
print("rho1 = {} kg/m^3".format(rho))
print("Inlet hub radius = {} m".format(rh_i))
print("Inlet tip radius = {} m".format(rt_i))
print("Area = {} m^2".format(A1))

Ut = N*2*np.pi*rt_i
print("Ut = {} m/s".format(Ut))

V1t = ((Ut**2)+(Ca**2)) ** 0.5
print("V1t = {} m/s".format(V1t))
print("a = {} m/s".format(a_inlet))
M1t = V1t / a_inlet
print("M1t = {}".format(M1t))

r_mean = (rt_i + rh_i) / 2
P02 = Comp_ratio * P01
print("P02 = {} P01".format(P02))

n = 1/(1 - (((gamma-1)/(polytropic_efficiency*gamma))))
print("(n-1)/n = {}".format((n-1)/n))
T02 = T01 * (Comp_ratio**((n-1)/n))
print("T02 = {} K".format(T02))

T2 = T02 - ((Ca**2)/(2*Cp))
P2 = P02*((T2/T02)**(gamma/(gamma-1)))
rho2 = P2/(R*T2)
print("T2 = {} K".format(T2))
print("P2 = {} P01".format(P2))
print("rho2 = {} kg/m^3".format(rho2))
A2 = massa_flow / (rho2 * Ca)
print("A2 = {} m^2".format(A2))

blade_height = A2 / (2*np.pi*r_mean)
print("blade height = {}".format(blade_height))
rt_o = r_mean + (blade_height/2)
rh_o = r_mean - (blade_height/2)
print("Outlet tip radius = {} m".format(rt_o))
print("Outlet hub radius = {} m".format(rh_o))

change_stagT = T02 - T01
print("Change in stagnation temperature = {} K".format(change_stagT))

######## Estimation of Stages
U = 2*np.pi*r_mean*N
print("U = {} m/s".format(U))
beta_1 = np.arctan(U / Ca)
print("Beta 1 = {} degrees".format((180/np.pi)*beta_1))
V1 = Ca / np.cos(beta_1)
print("V1 = {} m/s".format(V1))
#### De Haller Criterion
V2 = V1 * 0.72
beta_2 = np.arccos(Ca/V2)
print("Beta 2 = {} degrees".format((180/np.pi)*beta_2))
change_stagT_stage = (U*Ca*(np.tan(beta_1)-np.tan(beta_2)))/(Cp)
print("Approximate change in stagnation temperature per stage = {}".format(change_stagT_stage))
num_stages = change_stagT / change_stagT_stage
print("Approximate number of stages = {}".format(num_stages))

######## Stage-by-stage design ###################### COD
CD = {}
num_stages = 5
work_done_factors = [0.98,0.93,0.88,0.83,0.83]
degrees_of_reactions = [0.85,0.7,0.7,0.7,0.7]
temperature_rises = [44.75,38.75,37.25,37.1] # one less than num_stages defined and let the last stage take the remaining change.
temperature_rises += [change_stagT-sum(temperature_rises)]
stator_turning_angles = [44,39.5,39.25,39.5,39.5]
print("Stage work done factors = {}".format(work_done_factors))
print("Stage degrees of reaction = {}".format(degrees_of_reactions))
print("Stage temperature rises = {}".format(temperature_rises))
print("Frist stage stator turning angle = {}".format(stator_turning_angles))
for i in range(num_stages):
    CD[i+1] = {"WD":work_done_factors[i],"DR":degrees_of_reactions[i],"TR":temperature_rises[i],"STA":stator_turning_angles[i]}
for i in range(1,num_stages+1):
    print("\nStage {}:".format(i))
    if i == 1: # if the first stage
        Cw1 = 0
        results = COD_rot_stat(T01,P01,CD[i]["TR"],CD[i]["DR"],n,Ca,Cp,gamma,R,massa_flow,rt_i,N)
        U1,T01r,P01r,Rho1r,A1r,rh1r,U2,T02s,P02s,Rho2s,A2s,rh2s = results
        pressure_ratio = (1 + (polytropic_efficiency*CD[i]["TR"]/T01)) ** (gamma/(gamma-1))
        P0o,T0o = [P01 * pressure_ratio,T01 + CD[i]["TR"]]
    # elif i == 2: #num_stages:
    else:
        # Cw1 = np.tan((np.pi/180)*(CD[i-1]["alpha2"] - CD[i-1]["STA"]))*CD[i-1]["Cw2"]
        Cw1 = Ca * np.tan((np.pi/180)*(CD[i-1]["alpha3"]))
        results = COD_rot_stat(CD[i-1]["T0o"],CD[i-1]["P0o"],CD[i]["TR"],CD[i]["DR"],n,Ca,Cp,gamma,R,massa_flow,rt_i,N)
        U1,T01r,P01r,Rho1r,A1r,rh1r,U2,T02s,P02s,Rho2s,A2s,rh2s = results
        b1,b2 = symbols("b1 b2")
        # eq1 = ((tan(b1)**2) - (tan(b2)**2)) - (2*CD[i]["TR"]*Cp*CD[i]["DR"]/(Ca**2))
        # eq2 = - (Ca*U2*tan(b2)) + (U2**2) + (Ca*U1*tan(b1)) - (U1**2) - (CD[i]["TR"]*Cp/CD[i]["WD"])
        # beta1,beta2 = solve([eq1,eq2],[b1,b2])[0]
        # print((180/np.pi)*float(beta1),(180/np.pi)*float(beta2))
        pressure_ratio = (1 + (polytropic_efficiency*CD[i]["TR"]/CD[i-1]["T0o"])) ** (gamma/(gamma-1))
        P0o,T0o = [CD[i-1]["P0o"] * pressure_ratio,CD[i-1]["T0o"] + CD[i]["TR"]]
    CD[i]["Cw1"] = Cw1
    CD[i]["U1"],CD[i]["U2"],CD[i]["rhr_rt"],CD[i]["rhs_rt"] = [U1,U2,rh1r/rt_i,rh2s/rt_i]
    print("U1 rotor = {} m/s\nU2 stator = {} m/s\nrotor hub-to-tip ratio = {}\nstator hub-to-tip ratio = {}".format(U1,U2,CD[i]["rhr_rt"],CD[i]["rhs_rt"]))
    Cw2 = ((Cp*CD[i]["TR"]/CD[i]["WD"]) + (U1 * Cw1)) / U2
    beta_1,alpha_1 = [(180/np.pi)*np.arctan((U1-Cw1)/Ca),(180/np.pi)*np.arctan(Cw1/Ca)]
    beta_2,alpha_2 = [(180/np.pi)*np.arctan((U2-Cw2)/Ca),(180/np.pi)*np.arctan(Cw2/Ca)]
    print("Cw1 = {} m/s\nCw2 = {} m/s\nbeta 1 = {} deg.\nbeta 2 = {} deg.\nalpha 1 = {} deg.\nalpha 2 = {} deg.".format(Cw1,Cw2,beta_1,beta_2,alpha_1,alpha_2))
    CD[i]["Cw1"],CD[i]["Cw2"],CD[i]["beta1"],CD[i]["beta2"],CD[i]["alpha1"],CD[i]["alpha2"] = [Cw1,Cw2,beta_1,beta_2,alpha_1,alpha_2]
    CD[i]["alpha3"] = CD[i]["alpha2"] - CD[i]["STA"]
    print("alpha 3 = {} deg.".format(CD[i]["alpha3"]))
    RdeHaller = np.cos((np.pi/180)*beta_1)/np.cos((np.pi/180)*beta_2)
    SdeHaller = np.cos((np.pi/180)*CD[i]["alpha2"])/np.cos((np.pi/180)*(CD[i]["alpha3"]))
    CD[i]["RdeHaller"],CD[i]["SdeHaller"] = [RdeHaller,SdeHaller]
    print("Rotor de Haller number = {}".format(RdeHaller))
    print("Stator de Haller number = {}".format(SdeHaller))
    print("pressure ratio = {}\nOutlet pressure = {} Pa\nOutlet Temperature = {} K".format(pressure_ratio,P0o,T0o))
    CD[i]["PR"],CD[i]["P0o"],CD[i]["T0o"] = [pressure_ratio,P0o,T0o]
    # CD[i]["DR"] = 1 - ((Cw2+Cw1)/(2*U))
    CD[i]["DR"] = ((((Ca/np.cos((np.pi/180)*CD[i]["beta1"]))**2)-((Ca/np.cos((np.pi/180)*CD[i]["beta2"]))**2))/2)/((Cw2*U2)-(Cw1*U1))
    print("Estimated degree of reaction = {}".format(CD[i]["DR"]))
