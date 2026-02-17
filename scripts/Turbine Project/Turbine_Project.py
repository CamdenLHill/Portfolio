# Camden Hill 11/21/24

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def turbine_simple_ex(m_flow,turb_eff,T01,T03,P_ratio,P01,N,U,n_loss_coeff,Cp,gamma,R,flow_coeff,alpha_3,k_h2):
    rad_to_degree = (180/np.pi)
    degree_to_rad = (np.pi/180)
    alpha_1 = 0
    print("nozzle loss coefficient = {}".format(n_loss_coeff))
    dT = T01 - T03
    t_drop_coeff = (2*Cp*(dT)) / (U**2)
    print("Temperature Drop Coefficient = {}".format(t_drop_coeff))

    beta_3 = np.arctan((1/flow_coeff)+np.tan(degree_to_rad*alpha_3)) * rad_to_degree
    print("Flow Coefficient = {}".format(flow_coeff))

    degree_reaction = ((np.tan(degree_to_rad*beta_3)*2*flow_coeff)-(0.5*t_drop_coeff))/2
    print("Degree of Reaction = {}".format(degree_reaction))

    beta_2 = np.arctan((1/(2*flow_coeff))*((0.5*t_drop_coeff)-(2*degree_reaction))) * rad_to_degree
    alpha_2 = np.arctan(np.tan(beta_2*degree_to_rad)+(1/flow_coeff)) * rad_to_degree
    print("beta_2 = {} degrees".format(beta_2))
    print("alpha_2 = {} degrees".format(alpha_2))
    print("beta_3 = {} degrees".format(beta_3))
    print("alpha_3 = {} degrees".format(alpha_3))

    Ca2 = U*flow_coeff
    C2 = Ca2/np.cos(alpha_2*degree_to_rad)
    print("Ca2 = {} m/s".format(Ca2))
    print("C2 = {} m/s".format(C2))

    T02 = T01
    T2 = T02 - ((C2**2)/(2*Cp))
    T2_ = T2 - (n_loss_coeff*(C2**2)/(2*Cp))
    print("T02 = {} K".format(T02))
    print("T2 = {} K".format(T2))
    print("T2' = {} K".format(T2_))

    P2 = P01 / ((T01/T2_)**(gamma/(gamma-1)))
    P_real_ratio = P01 / P2
    P_crit_ratio = (((gamma+1)/2)**(gamma/(gamma-1)))
    print("P2 = {} Pa".format(P2))
    print("p_real_ratio = {}".format(P_real_ratio))
    print("P_crit_ratio = {}".format(P_crit_ratio))
    print("P_real_ratio < P_crit_ratio = {}".format(P_real_ratio < P_crit_ratio))

    rho2 = P2 / (R*T2)
    A2 = m_flow/(rho2*Ca2)
    A2N = m_flow/(rho2*C2)
    print("rho2 = {} kg/m^3".format(rho2))
    print("Area 2 = {} m^2".format(A2))
    print("Area 2 Nozzle = {} m^2".format(A2N))

    Ca3 = Ca2
    C3 = Ca3 / np.cos(alpha_3*degree_to_rad)
    Ca1 = C3
    C1 = Ca1
    print("Ca3 = {} m/s".format(C3))
    print("C3 = {} m/s".format(C3))
    print("Ca1 = {} m/s".format(Ca1))
    print("C1 = {} m/s".format(C1))

    T1 = T01 - ((C1**2)/(2*Cp))
    P1 = P01 * ((T1/T01)**(gamma/(gamma-1)))
    rho1 = P1 / (R*T1)
    A1 = m_flow/(rho1*Ca1)
    print("T1 = {} K".format(T1))
    print("P1 = {} Pa".format(P1))
    print("rho1 = {} kg/m^3".format(rho1))
    print("Area 1 = {} m^2".format(A1))

    T3 = T03 - ((C3**2)/(2*Cp))
    P03 = P01 / P_ratio
    P3 = P03*((T3/T03)**(gamma/(gamma-1)))
    rho3 = P3 / (R*T3)
    A3 = m_flow/(rho3*Ca3)
    print("T3 = {} K".format(T3))
    print("P03 = {} Pa".format(P03))
    print("P3 = {} Pa".format(P3))
    print("rho3 = {} kg/m^3".format(rho3))
    print("Area 3 = {} m^2".format(A3))

    rm = U / (2*np.pi*N)
    h1 = A1*N/U
    h2 = A2*N/U
    h3 = A3*N/U
    rt_rr1 = (rm+(h1/2)) / (rm-(h1/2))
    rt_rr2 = (rm+(h2/2)) / (rm-(h2/2))
    rt_rr3 = (rm+(h3/2)) / (rm-(h3/2))
    print("A1 = {} m^2".format(A1))
    print("h1 = {} m".format(h1))
    print("rt/rr 1 = {}".format(rt_rr1))
    print("A2 = {} m^2".format(A2))
    print("h2 = {} m".format(h2))
    print("rt/rr 2 = {}".format(rt_rr2))
    print("A3 = {} m^2".format(A3))
    print("h3 = {} m".format(h3))
    print("rt/rr 3 = {}".format(rt_rr3))

    T3__ = T2 / ((P2/P3)**((gamma-1)/gamma))
    V3 = Ca3 / np.cos(beta_3*degree_to_rad)
    r_loss_coeff = (T3-T3__) / ((V3**2)/(2*Cp))
    print("T3'' = {} K".format(T3__))
    print("V3 = {} m/s".format(V3))
    print("rotor loss coefficient = {}".format(r_loss_coeff))

    rm_rr2 = rm / (rm-(h2/2))
    rm_rt2 = rm / (rm+(h2/2))
    rm_rr3 = rm / (rm-(h3/2))
    rm_rt3 = rm / (rm+(h3/2))
    print("(rm/rr)_2 = {}".format(rm_rr2))
    print("(rm/rt)_2 = {}".format(rm_rt2))
    print("(rm/rr)_3 = {}".format(rm_rr3))
    print("(rm/rt)_3 = {}".format(rm_rt3))

    alpha_2r = np.arctan(rm_rr2*np.tan(alpha_2*degree_to_rad))*rad_to_degree
    alpha_3r = np.arctan(rm_rr3*np.tan(alpha_3*degree_to_rad))*rad_to_degree
    beta_2r = np.arctan(np.tan(alpha_2r*degree_to_rad)-((1/rm_rr2)*(U/Ca2)))*rad_to_degree
    beta_3r = np.arctan(np.tan(alpha_3r*degree_to_rad)+((1/rm_rr3)*(U/Ca3)))*rad_to_degree
    print("Root:")
    print("alpha_2r = {} degrees".format(alpha_2r))
    print("alpha_3r = {} degrees".format(alpha_3r))
    print("beta_2r = {} degrees".format(beta_2r))
    print("beta_3r = {} degrees".format(beta_3r))

    print("alpha_2m = {} degrees".format(alpha_2))
    print("alpha_3m = {} degrees".format(alpha_3))
    print("beta_2m = {} degrees".format(beta_2))
    print("beta_3m = {} degrees".format(beta_3))

    alpha_2t = np.arctan(rm_rt2*np.tan(alpha_2*degree_to_rad))*rad_to_degree
    alpha_3t = np.arctan(rm_rt3*np.tan(alpha_3*degree_to_rad))*rad_to_degree
    beta_2t = np.arctan(np.tan(alpha_2t*degree_to_rad)-((1/rm_rt2)*(U/Ca2)))*rad_to_degree
    if beta_2t < 0:
        if abs(beta_2t) < 1: ### Essentially just rounding
            beta_2t = 0
    beta_3t = np.arctan(np.tan(alpha_3t*degree_to_rad)+((1/rm_rt3)*(U/Ca3)))*rad_to_degree
    print("Tip:")
    print("alpha_2t = {} degrees".format(alpha_2t))
    print("alpha_3t = {} degrees".format(alpha_3t))
    print("beta_2t = {} degrees".format(beta_2t))
    print("beta_3t = {} degrees".format(beta_3t))
    
    V2r = Ca2/np.cos(beta_2r*degree_to_rad)
    C2r = Ca2/np.cos(alpha_2r*degree_to_rad)
    T2r = T02 - ((C2r**2)/(2*Cp))
    MV2r = V2r / ((gamma*R*T2r)**0.5)
    print("Checking Root Mach Number:")
    print("V2r = {} m/s".format(V2r))
    print("C2r = {} m/s".format(C2r))
    print("T2r = {} K".format(T2r))
    print("(MV2)r = {}".format(MV2r))
    print("(MV2)r < 0.75: {}".format(MV2r<0.75))

    s_cN = 0.86
    s_cR = 0.83
    print("s_cN = {}".format(s_cN))
    print("s_cR = {}".format(s_cR))
    h_N = 0.5*(h1+h2)
    h_R = 0.5*(h2+h3)
    print("h_N = {} m".format(h_N))
    print("h_R = {} m".format(h_R))

    h_c = 3
    c_N = h_N/h_c
    c_R = h_R/h_c
    s_N = s_cN*c_N
    s_R = s_cR*c_R
    num_bladesN = int((2*np.pi*rm/s_N)+1)
    num_bladesR = int((2*np.pi*rm/s_R)+1)
    print("c_N = {} m".format(c_N))
    print("c_R = {} m".format(c_R))
    print("s_N = {} m".format(s_N))
    print("s_R = {} m".format(s_R))
    print("Number of Nozzle Blades = {}".format(num_bladesN))
    print("Number of Rotor Blades = {}".format(num_bladesR))

    blade_metal_density = 8000
    A_avg = 0.5*(A2+A3)
    max_stress = (4/3)*np.pi*(N**2)*blade_metal_density*A_avg
    print("Max centrifgual tensile stress = {} MN/m^2".format(max_stress/(10**6)))

    t_c = 0.2
    Yp_b2_0 = 0.023
    Yp_b2_b3 = 0.087
    YpN = 0.024
    YpR = (Yp_b2_0 + (((beta_2/beta_3)**2)*(Yp_b2_b3-Yp_b2_0)))*((t_c/0.2)**(beta_2/beta_3))
    print("YpN = {}".format(YpN))
    print("YpR = {}".format(YpR))

    beta_m = np.arctan((np.tan(beta_3*degree_to_rad)-np.tan(beta_2*degree_to_rad))/2) * rad_to_degree
    print("beta_m = {} degrees".format(beta_m))
    
    k_h1 = 0 # fully shrouded and no leakage in nozzle
    # k_h2 = 0.02 # clearance % of blade height
    B1 = 0 # fully shrouded and no leakage in nozzle
    B2 = 0.5 # 
    secondary_loss_parameter1 = (((A2*np.cos(alpha_2*degree_to_rad))/(A1*np.cos(alpha_1*degree_to_rad)))**2) / (1+(1/((rt_rr1+rt_rr2)/2)))
    lambda_1 = 0.012
    alpha_m = np.arctan((np.tan(alpha_2*degree_to_rad)-np.tan(alpha_1*degree_to_rad))/2) * rad_to_degree
    CL_s_cN = 2*(np.tan(alpha_1*degree_to_rad)+np.tan(alpha_2*degree_to_rad))*np.cos(alpha_m*degree_to_rad)
    Ys_YkN = (lambda_1+(B1*k_h1))*(CL_s_cN**2)*((np.cos(alpha_2*degree_to_rad)**2)/(np.cos(alpha_m*degree_to_rad)**3))
    print("secondary_loss_parameter1 = {}".format(secondary_loss_parameter1))
    print("second loss coefficient 2 = {}".format(lambda_1))
    print("alpha_m = {} degrees".format(alpha_m))
    print("CL_s_cN = {}".format(CL_s_cN))
    print("Ys_YkN = {}".format(Ys_YkN))


    lambda_2 = 0.015
    secondary_loss_parameter2 = (((A3*np.cos(beta_3*degree_to_rad))/(A2*np.cos(beta_2*degree_to_rad)))**2) / (1+(1/((rt_rr2+rt_rr3)/2)))
    CL_s_cR = 2*(np.tan(beta_3*degree_to_rad)+np.tan(beta_2*degree_to_rad))*np.cos(beta_m*degree_to_rad)
    Ys_YkR = (lambda_2+(B2*k_h2))*(CL_s_cR**2)*((np.cos(beta_3*degree_to_rad)**2)/(np.cos(beta_m*degree_to_rad)**3))
    print("secondary_loss_parameter2 = {}".format(secondary_loss_parameter2))
    print("second loss coefficient 2 = {}".format(lambda_2))
    print("beta_m = {} degrees".format(beta_m))
    print("CL_s_cR = {}".format(CL_s_cR))
    print("Ys_YkR = {}".format(Ys_YkR))

    YN = YpN + Ys_YkN
    YR = YpR + Ys_YkR
    print("YN = {}".format(YN))
    print("YR = {}".format(YR))

    n_loss_coeff = YN/(T02/T2_)
    T03rel = T3 + ((V3**2)/(2*Cp))
    r_loss_coeff = YR/(T03rel/T3__)
    print("n_loss_coeff = {}".format(n_loss_coeff))
    print("r_loss_coeff = {}".format(r_loss_coeff))

    Ca = Ca2 # Ca3
    eff_calc1_num = ((r_loss_coeff)/(np.cos(beta_3*degree_to_rad)**2)) + ((T3/T2)*n_loss_coeff/(np.cos(alpha_2*degree_to_rad)**2))
    eff_calc1_den = np.tan(beta_3*degree_to_rad) + np.tan(alpha_2*degree_to_rad) - (U/Ca)
    turb_eff = 1/(1+((Ca/(2*U))*(eff_calc1_num/eff_calc1_den)))
    print("Turbine Isentropic Efficiency = {}".format(turb_eff))

def turbine_simple_problem(m_flow,turb_eff,T01,T03,P_ratio,P01,N,U,n_loss_coeff,Cp,gamma,R,flow_coeff,alpha_3,k_h2):
    rad_to_degree = (180/np.pi)
    degree_to_rad = (np.pi/180)
    alpha_1 = 0
    print("nozzle loss coefficient = {}".format(n_loss_coeff))
    dT = T01 - T03
    t_drop_coeff = (2*Cp*(dT)) / (U**2)
    print("Temperature Drop = {} K".format(dT))
    print("Temperature Drop Coefficient = {}".format(t_drop_coeff))

    beta_3 = np.arctan((1/flow_coeff)+np.tan(degree_to_rad*alpha_3)) * rad_to_degree
    print("Flow Coefficient = {}".format(flow_coeff))

    degree_reaction = ((np.tan(degree_to_rad*beta_3)*2*flow_coeff)-(0.5*t_drop_coeff))/2
    print("Degree of Reaction = {}".format(degree_reaction))

    beta_2 = np.arctan((1/(2*flow_coeff))*((0.5*t_drop_coeff)-(2*degree_reaction))) * rad_to_degree
    alpha_2 = np.arctan(np.tan(beta_2*degree_to_rad)+(1/flow_coeff)) * rad_to_degree
    print("beta_2 = {} degrees".format(beta_2))
    print("alpha_2 = {} degrees".format(alpha_2))
    print("beta_3 = {} degrees".format(beta_3))
    print("alpha_3 = {} degrees".format(alpha_3))

    Ca2 = U*flow_coeff
    C2 = Ca2/np.cos(alpha_2*degree_to_rad)
    print("Ca2 = {} m/s".format(Ca2))
    print("C2 = {} m/s".format(C2))

    T02 = T01
    T2 = T02 - ((C2**2)/(2*Cp))
    T2_ = T2 - (n_loss_coeff*(C2**2)/(2*Cp))
    print("T02 = {} K".format(T02))
    print("T2 = {} K".format(T2))
    print("T2' = {} K".format(T2_))

    P2 = P01 / ((T01/T2_)**(gamma/(gamma-1)))
    P_real_ratio = P01 / P2
    P_crit_ratio = (((gamma+1)/2)**(gamma/(gamma-1)))
    print("P2 = {} Pa".format(P2))
    print("p_real_ratio = {}".format(P_real_ratio))
    print("P_crit_ratio = {}".format(P_crit_ratio))
    print("P_real_ratio < P_crit_ratio = {}".format(P_real_ratio < P_crit_ratio))

    rho2 = P2 / (R*T2)
    A2 = m_flow/(rho2*Ca2)
    A2N = m_flow/(rho2*C2)
    print("rho2 = {} kg/m^3".format(rho2))
    print("Area 2 = {} m^2".format(A2))
    print("Area 2 Nozzle = {} m^2".format(A2N))

    Ca3 = Ca2
    C3 = Ca3 / np.cos(alpha_3*degree_to_rad)
    Ca1 = C3
    C1 = Ca1
    print("Ca3 = {} m/s".format(C3))
    print("C3 = {} m/s".format(C3))
    print("Ca1 = {} m/s".format(Ca1))
    print("C1 = {} m/s".format(C1))

    T1 = T01 - ((C1**2)/(2*Cp))
    P1 = P01 * ((T1/T01)**(gamma/(gamma-1)))
    rho1 = P1 / (R*T1)
    A1 = m_flow/(rho1*Ca1)
    print("T1 = {} K".format(T1))
    print("P1 = {} Pa".format(P1))
    print("rho1 = {} kg/m^3".format(rho1))
    print("Area 1 = {} m^2".format(A1))

    T3 = T03 - ((C3**2)/(2*Cp))
    P03 = P01 / P_ratio
    P3 = P03*((T3/T03)**(gamma/(gamma-1)))
    rho3 = P3 / (R*T3)
    A3 = m_flow/(rho3*Ca3)
    print("T3 = {} K".format(T3))
    print("P03 = {} Pa".format(P03))
    print("P3 = {} Pa".format(P3))
    print("rho3 = {} kg/m^3".format(rho3))
    print("Area 3 = {} m^2".format(A3))

    rm = U / (2*np.pi*N)
    h1 = A1*N/U
    h2 = A2*N/U
    h3 = A3*N/U
    rt1,rr1 = [(rm+(h1/2)),(rm-(h1/2))]
    rt2,rr2 = [(rm+(h2/2)),(rm-(h2/2))]
    rt3,rr3 = [(rm+(h3/2)),(rm-(h3/2))]
    rt_rr1 = rt1/rr1
    rt_rr2 = rt2/rr2
    rt_rr3 = rt3/rr3
    print("A1 = {} m^2".format(A1))
    print("h1 = {} m".format(h1))
    print("rt/rr 1 = {}".format(rt_rr1))
    print("A2 = {} m^2".format(A2))
    print("h2 = {} m".format(h2))
    print("rt/rr 2 = {}".format(rt_rr2))
    print("A3 = {} m^2".format(A3))
    print("h3 = {} m".format(h3))
    print("rt/rr 3 = {}".format(rt_rr3))

    T3__ = T2 / ((P2/P3)**((gamma-1)/gamma))
    V3 = Ca3 / np.cos(beta_3*degree_to_rad)
    r_loss_coeff = (T3-T3__) / ((V3**2)/(2*Cp))
    print("T3'' = {} K".format(T3__))
    print("V3 = {} m/s".format(V3))
    print("rotor loss coefficient = {}".format(r_loss_coeff))

    rm_rr2 = rm / (rm-(h2/2))
    rm_rt2 = rm / (rm+(h2/2))
    rm_rr3 = rm / (rm-(h3/2))
    rm_rt3 = rm / (rm+(h3/2))
    print("(rm/rr)_2 = {}".format(rm_rr2))
    print("(rm/rt)_2 = {}".format(rm_rt2))
    print("(rm/rr)_3 = {}".format(rm_rr3))
    print("(rm/rt)_3 = {}".format(rm_rt3))

    alpha_2r = np.arctan(rm_rr2*np.tan(alpha_2*degree_to_rad))*rad_to_degree
    alpha_3r = np.arctan(rm_rr3*np.tan(alpha_3*degree_to_rad))*rad_to_degree
    beta_2r = np.arctan(np.tan(alpha_2r*degree_to_rad)-((1/rm_rr2)*(U/Ca2)))*rad_to_degree
    beta_3r = np.arctan(np.tan(alpha_3r*degree_to_rad)+((1/rm_rr3)*(U/Ca3)))*rad_to_degree
    print("Root:")
    print("alpha_2r = {} degrees".format(alpha_2r))
    print("alpha_3r = {} degrees".format(alpha_3r))
    print("beta_2r = {} degrees".format(beta_2r))
    print("beta_3r = {} degrees".format(beta_3r))

    print("alpha_2m = {} degrees".format(alpha_2))
    print("alpha_3m = {} degrees".format(alpha_3))
    print("beta_2m = {} degrees".format(beta_2))
    print("beta_3m = {} degrees".format(beta_3))

    alpha_2t = np.arctan(rm_rt2*np.tan(alpha_2*degree_to_rad))*rad_to_degree
    alpha_3t = np.arctan(rm_rt3*np.tan(alpha_3*degree_to_rad))*rad_to_degree
    beta_2t = np.arctan(np.tan(alpha_2t*degree_to_rad)-((1/rm_rt2)*(U/Ca2)))*rad_to_degree
    if beta_2t < 0:
        if abs(beta_2t) < 1: ### Essentially just rounding
            beta_2t = 0
    beta_3t = np.arctan(np.tan(alpha_3t*degree_to_rad)+((1/rm_rt3)*(U/Ca3)))*rad_to_degree
    print("Tip:")
    print("alpha_2t = {} degrees".format(alpha_2t))
    print("alpha_3t = {} degrees".format(alpha_3t))
    print("beta_2t = {} degrees".format(beta_2t))
    print("beta_3t = {} degrees".format(beta_3t))
    
    V2r = Ca2/np.cos(beta_2r*degree_to_rad)
    C2r = Ca2/np.cos(alpha_2r*degree_to_rad)
    T2r = T02 - ((C2r**2)/(2*Cp))
    MV2r = V2r / ((gamma*R*T2r)**0.5)
    print("Checking Root Mach Number:")
    print("V2r = {} m/s".format(V2r))
    print("C2r = {} m/s".format(C2r))
    print("T2r = {} K".format(T2r))
    print("(MV2)r = {}".format(MV2r))
    print("(MV2)r < 0.75: {}".format(MV2r<0.75))
    ######################################################################
    print("\nGraph 1:")
    ######################################################################
    s_cN = 0.91
    s_cR = 0.889
    print("s_cN = {}".format(s_cN))
    print("s_cR = {}".format(s_cR))
    h_N = 0.5*(h1+h2)
    h_R = 0.5*(h2+h3)
    print("h_N = {} m".format(h_N))
    print("h_R = {} m".format(h_R))

    h_c = 3
    c_N = h_N/h_c
    c_R = h_R/h_c
    s_N = s_cN*c_N
    s_R = s_cR*c_R
    num_bladesN = int((2*np.pi*rm/s_N)+1)
    num_bladesR = int((2*np.pi*rm/s_R)+1)
    print("c_N = {} m".format(c_N))
    print("c_R = {} m".format(c_R))
    print("s_N = {} m".format(s_N))
    print("s_R = {} m".format(s_R))
    print("Number of Nozzle Blades = {}".format(num_bladesN))
    print("Number of Rotor Blades = {}".format(num_bladesR))

    blade_metal_density = 8000
    A_avg = 0.5*(A2+A3)
    max_stress = (4/3)*np.pi*(N**2)*blade_metal_density*A_avg
    print("Max centrifgual tensile stress = {} MPa".format(max_stress/(10**6)))
    ######################################################################
    print("\nGraph 2:")
    ######################################################################
    t_c = 0.2
    Yp_b2_0 = 0.021
    Yp_b2_b3 = 0.084
    YpN = 0.02
    YpR = (Yp_b2_0 + (((beta_2/beta_3)**2)*(Yp_b2_b3-Yp_b2_0)))*((t_c/0.2)**(beta_2/beta_3))
    print("YpN = {}".format(YpN))
    print("YpR = {}".format(YpR))

    beta_m = np.arctan((np.tan(beta_3*degree_to_rad)-np.tan(beta_2*degree_to_rad))/2) * rad_to_degree
    print("beta_m = {} degrees".format(beta_m))
    ######################################################################
    print("\nGraph 3:")
    ######################################################################
    k_h1 = 0 # fully shrouded and no leakage in nozzle
    # k_h2 = 0.02 # clearance % of blade height
    B1 = 0 # fully shrouded and no leakage in nozzle
    B2 = 0.5 # 
    secondary_loss_parameter1 = (((A2*np.cos(alpha_2*degree_to_rad))/(A1*np.cos(alpha_1*degree_to_rad)))**2) / (1+(1/((rt_rr1+rt_rr2)/2)))
    lambda_1 = 0.019
    alpha_m = np.arctan((np.tan(alpha_2*degree_to_rad)-np.tan(alpha_1*degree_to_rad))/2) * rad_to_degree
    CL_s_cN = 2*(np.tan(alpha_1*degree_to_rad)+np.tan(alpha_2*degree_to_rad))*np.cos(alpha_m*degree_to_rad)
    Ys_YkN = (lambda_1+(B1*k_h1))*(CL_s_cN**2)*((np.cos(alpha_2*degree_to_rad)**2)/(np.cos(alpha_m*degree_to_rad)**3))
    print("secondary_loss_parameter1 = {}".format(secondary_loss_parameter1))
    print("second loss coefficient 1 = {}".format(lambda_1))
    print("alpha_m = {} degrees".format(alpha_m))
    print("CL_s_cN = {}".format(CL_s_cN))
    print("Ys_YkN = {}".format(Ys_YkN))


    lambda_2 = 0.0232
    secondary_loss_parameter2 = (((A3*np.cos(beta_3*degree_to_rad))/(A2*np.cos(beta_2*degree_to_rad)))**2) / (1+(1/((rt_rr2+rt_rr3)/2)))
    CL_s_cR = 2*(np.tan(beta_3*degree_to_rad)+np.tan(beta_2*degree_to_rad))*np.cos(beta_m*degree_to_rad)
    Ys_YkR = (lambda_2+(B2*k_h2))*(CL_s_cR**2)*((np.cos(beta_3*degree_to_rad)**2)/(np.cos(beta_m*degree_to_rad)**3))
    print("secondary_loss_parameter2 = {}".format(secondary_loss_parameter2))
    print("second loss coefficient 2 = {}".format(lambda_2))
    print("beta_m = {} degrees".format(beta_m))
    print("CL_s_cR = {}".format(CL_s_cR))
    print("Ys_YkR = {}".format(Ys_YkR))

    YN = YpN + Ys_YkN
    YR = YpR + Ys_YkR
    print("YN = {}".format(YN))
    print("YR = {}".format(YR))

    n_loss_coeff = YN/(T02/T2_)
    T03rel = T3 + ((V3**2)/(2*Cp))
    r_loss_coeff = YR/(T03rel/T3__)
    print("n_loss_coeff = {}".format(n_loss_coeff))
    print("r_loss_coeff = {}".format(r_loss_coeff))

    Ca = Ca2 # Ca3
    eff_calc1_num = ((r_loss_coeff)/(np.cos(beta_3*degree_to_rad)**2)) + ((T3/T2)*n_loss_coeff/(np.cos(alpha_2*degree_to_rad)**2))
    eff_calc1_den = np.tan(beta_3*degree_to_rad) + np.tan(alpha_2*degree_to_rad) - (U/Ca)
    turb_eff = 1/(1+((Ca/(2*U))*(eff_calc1_num/eff_calc1_den)))
    print("Turbine Isentropic Efficiency = {}".format(turb_eff))


    SSD = {"(MV2)r":[MV2r],
           "P critical ratio":[P_crit_ratio],
           "P real ratio":[P_real_ratio],
           "Max stress (MPa)":[max_stress/(10**6)]}
    print((pd.DataFrame(SSD).T).to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))

    # Stations
    # Area blade height tip-to-root ratio
    BSD = {"A (m^2)":[A1,A2,A3],
           "h (m)":[h1,h2,h3],
           "rt_rr":[rt_rr1,rt_rr2,rt_rr3]} # Blade sizing dictionary
    print((pd.DataFrame(BSD).T).to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
    
    # Gas Angles
    # root mean tip
    # alpha_1 alpha_2 alpha_3 beta_2 beta_3
    AD = {"Root":[0,alpha_2r,alpha_3r,beta_2r,beta_3r],
          "Mean":[0,alpha_2,alpha_3,beta_2,beta_3],
          "Tip":[0,alpha_2t,alpha_3t,beta_2t,beta_3t]}
    print((pd.DataFrame(AD).T).to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
    
    # Plane static variables
    # P T rho
    # 1 2 3
    PD = {"Pressure":[P1/(10**5),P2/(10**5),P3/(10**5)],
          "Temperature":[T1,T2,T3],
          "Density":[rho1,rho2,rho3]}
    print((pd.DataFrame(PD).T).to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
    # Blade Row
    # Nozzle Rotor
    # s/c h (mean) h/c c s n
    BD = {"s/c":[s_cN,s_cR],
          "h (mean)":[h_N,h_R],
          "h/c":[h_c,h_c],
          "c":[c_N,c_R],
          "s":[s_N,s_R],
          "# stages":[num_bladesN,num_bladesR]} # Blade dictionary
    print((pd.DataFrame(BD).T).to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
    
    YD = {"k_h":[k_h1,k_h2],
          "B":[B1,B2],
          "Loss Param.":[lambda_1,lambda_2],
          "Yp":[YpN,YpR],
          "CL/(s/c)":[CL_s_cN,CL_s_cR],
          "Y_s+Y_k":[Ys_YkN,Ys_YkR],
          "Y":[YN,YR]} # Loss Coefficient dictionary
    print((pd.DataFrame(YD).T).to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
    
    ED = {"Nozzle loss Coefficient":[n_loss_coeff],
          "Rotor loss Coefficient":[r_loss_coeff],
          "Isentropic Efficiency":[turb_eff]}
    print((pd.DataFrame(ED).T).to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))

    tips = [rt1,rt2,rt3]
    roots = [rr1,rr2,rr3]
    means = [rm,rm,rm]
    xvals = [1,2,3]

    plt.plot(xvals,tips,color=(np.random.random(),np.random.random(),np.random.random()),label="Tip Radii")
    plt.plot(xvals,roots,color=(np.random.random(),np.random.random(),np.random.random()),label="Root Radii")
    plt.plot(xvals,means,color=(np.random.random(),np.random.random(),np.random.random()),label="Mean Radii")
    plt.xticks(xvals)
    plt.grid()
    plt.legend()
    plt.xlabel("Station")
    plt.ylabel("length (m)")
    plt.title("Constant Mean Radius Turbine Stations")
    plt.show()
    print()
    # return 

# Example
m_flow = 20 # kg/s
turb_eff = 0.9
T01 = 1100 # K
T03 = T01 - 145 # K
P_ratio = 1.873
P01 = 4 * (10**5) # Pa
N = 250 # rev/s
U = 340 # m/s
rm = U/(2*np.pi*N)
n_loss_coeff = 0.05
Cp = 1.148 * (10**3)
gamma = 4/3
R = 287
flow_coeff = 0.8
alpha_3 = 10
k_h2 = 0.02

# turbine_simple_ex(m_flow,turb_eff,T01,T03,P_ratio,P01,N,U,n_loss_coeff,Cp,gamma,R,flow_coeff,alpha_3,k_h2)

# project
m_flow = 8 # kg/s
turb_eff = 0.88
T01 = 1173 # K
P_ratio = 2.16
Cp = 1.148 * (10**3)
gamma = 4/3
R = 287
# PT = 1.7 * (10**6)
# WT = PT / m_flow
h03 = T01 * Cp
h04i = Cp*(T01/((P_ratio)**((gamma-1)/(gamma))))
h04 = h03 - (turb_eff*(h03-h04i))
T03 = h04/Cp
P01 = 476.3 * (10**3) # Pa
N = 28150/60 # rev/s
print("N = {} rev/s".format(N))
U = (2*np.pi*N*rm) # m/s ###### Using the example rm
print("U = {} m/s".format(U))
U = 420
n_loss_coeff = 0.05
flow_coeff = 0.9
alpha_3 = 10
k_h2 = 0.01

turbine_simple_problem(m_flow,turb_eff,T01,T03,P_ratio,P01,N,U,n_loss_coeff,Cp,gamma,R,flow_coeff,alpha_3,k_h2)