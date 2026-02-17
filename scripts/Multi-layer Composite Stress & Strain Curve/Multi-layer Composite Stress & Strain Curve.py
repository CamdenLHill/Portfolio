# Camden Hill
# Composites Layering Code

import numpy as np
import matplotlib.pyplot as plt

def Q_bar_matrix(theta,Q11,Q12,Q22,Q66):
    T1 = np.array([[np.cos(theta)**2,np.sin(theta)**2,2*np.sin(theta)*np.cos(theta)],
                   [np.sin(theta)**2,np.cos(theta)**2,-2*np.sin(theta)*np.cos(theta)],
                   [-np.sin(theta)*np.cos(theta),np.sin(theta)*np.cos(theta),(np.cos(theta)**2)-(np.sin(theta)**2)]])
    T1_ = np.linalg.inv(T1)
    T2 = np.array([[np.cos(theta)**2,np.sin(theta)**2,np.sin(theta)*np.cos(theta)],
                   [np.sin(theta)**2,np.cos(theta)**2,-np.sin(theta)*np.cos(theta)],
                   [-2*np.sin(theta)*np.cos(theta),2*np.sin(theta)*np.cos(theta),(np.cos(theta)**2)-(np.sin(theta)**2)]])
    Q = np.array([[Q11,Q12,0],[Q12,Q22,0],[0,0,Q66]])
    Q_1 = np.matmul(T1_,np.matmul(Q,T2))
    # Q_11 = (Q11*(np.cos(theta)**4)) + (Q22*(np.sin(theta)**4)) + (2*(Q12+(2*Q66))*(np.sin(theta)**2)*(np.cos(theta)**2))
    # Q_22 = (Q11*(np.sin(theta)**4)) + (Q22*(np.cos(theta)**4)) + (2*(Q12+(2*Q66))*(np.sin(theta)**2)*(np.cos(theta)**2))
    # Q_12 = ((Q11+Q22-(4*Q66))*(np.sin(theta)**2)*(np.cos(theta)**2)) + (Q12*((np.cos(theta)**4)+(np.sin(theta)**4)))
    # Q_66 = ((Q11+Q22-(2*Q12)-(2*Q66))*(np.sin(theta)**2)*(np.cos(theta)**2)) + (Q66*((np.cos(theta)**4)+(np.sin(theta)**4)))
    # Q_16 = ((Q11-Q12-(2*Q66))*(np.cos(theta)**3)*np.sin(theta)) - ((Q22-Q12-(2*Q66))*np.cos(theta)*(np.sin(theta)**3))
    # Q_26 = ((Q11-Q12-(2*Q66))*np.cos(theta)*(np.sin(theta)**3)) - ((Q22-Q12-(2*Q66))*(np.cos(theta)**3)*np.sin(theta))
    # Q_2 = np.array([[Q_11,Q_12,Q_16],[Q_12,Q_22,Q_26],[Q_16,Q_26,Q_66]])
    # print(Q_2)
    return Q_1

def A_matrix_assembly(midpoint_layer_heights,layer_Qbars,layers): # mm's & GPa's
    A_matrix,inc = [np.array([[0,0,0],[0,0,0],[0,0,0]],dtype='float64'),0]
    for i in range(len(midpoint_layer_heights)-1):
        if i+1 in layers:
            A_matrix += (((midpoint_layer_heights[i+1])-(midpoint_layer_heights[i]))) * layer_Qbars[i-inc]
        else:
            inc += 1
    A_matrix = matrix_zeroing(A_matrix)
    print("A_matrix = \n{} GPa.mm".format(A_matrix))
    return A_matrix

def B_matrix_assembly(midpoint_layer_heights,layer_Qbars,layers):
    B_matrix,inc = [np.array([[0,0,0],[0,0,0],[0,0,0]],dtype='float64'),0]
    for i in range(len(midpoint_layer_heights)-1):
        if i+1 in layers:
            B_matrix += (0.5*((midpoint_layer_heights[i+1]**2)-(midpoint_layer_heights[i]**2))) * layer_Qbars[i-inc]
        else:
            inc += 1
    B_matrix = matrix_zeroing(B_matrix)
    print("B_matrix = \n{} GPa.mm^2".format(B_matrix))
    return B_matrix

def D_matrix_assembly(midpoint_layer_heights,layer_Qbars,layers):
    D_matrix,inc = [np.array([[0,0,0],[0,0,0],[0,0,0]],dtype='float64'),0]
    for i in range(len(midpoint_layer_heights)-1):
        if i+1 in layers:
            D_matrix += ((1/3)*((midpoint_layer_heights[i+1]**3)-(midpoint_layer_heights[i]**3))) * layer_Qbars[i-inc]
        else:
            inc += 1
    D_matrix = matrix_zeroing(D_matrix)
    print("D_matrix = \n{} GPa.mm^3".format(D_matrix))
    return D_matrix

def inverse_matrix_assembly(A_matrix,B_matrix,D_matrix):
    A_inv = np.linalg.inv(A_matrix)
    D_star = D_matrix - np.matmul(B_matrix,np.matmul(A_inv,B_matrix))
    D_prime = np.linalg.inv(D_star)
    A_prime = A_inv + np.matmul(A_inv,np.matmul(B_matrix,np.matmul(D_prime,np.matmul(B_matrix,A_inv))))
    B_prime = -np.matmul(A_inv,np.matmul(B_matrix,D_prime))
    C_prime = np.transpose(B_prime)
    return A_prime,B_prime,C_prime,D_prime

def laminate_stresses(strain_x0,strain_y0,strain_xy0,kx,ky,kxy,layer_Qbars,midpoint_layer_heights):
    stresses = []
    for i in range(len(midpoint_layer_heights)-1):
        stress = np.matmul(layer_Qbars[i],np.array([strain_x0,strain_y0,strain_xy0])) + (((midpoint_layer_heights[i]+midpoint_layer_heights[i+1])/2)*np.matmul(layer_Qbars[i],np.array([kx,ky,kxy])))
        stresses.append(stress)
        print("x-stress = {} GPa, y-stress = {} GPa, xy-shear stress = {} GPa".format(stress[0],stress[1],stress[2]))
    return stresses

def laminate_constitutive_relations(strain_x0,strain_y0,strain_xy0,kx,ky,kxy,A,B,D):
    large_matrix = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],dtype='float64')
    large_matrix[0:3,0:3] += A
    large_matrix[0:3,3:6] += B
    large_matrix[3:6,0:3] += B
    large_matrix[3:6,3:6] += D
    print("resultant matrix = \n{}".format(large_matrix))
    vector = np.array([strain_x0,strain_y0,strain_xy0,kx,ky,kxy])
    Nx,Ny,Nxy,Mx,My,Mxy = np.matmul(large_matrix,vector)
    print("Nx = {} GigaNewton-millimeters\nNy = {} GigaNewton-millimeters\nNxy = {} GigaNewton-millimeters\nMx = {} GigaNewton-millimeters^2\nMy = {} GigaNewton-millimeters^2\nMxy = {} GigaNewton-millimeters^2\n".format(Nx,Ny,Nxy,Mx,My,Mxy))
    Nx_,Ny_,Nxy_,Mx_,My_,Mxy_ = [Nx*(10**6),Ny*(10**6),Nxy*(10**6),Mx*(10**3),My*(10**3),Mxy*(10**3)]
    print("Nx = {} Newton-meters\nNy = {} Newton-meters\nNxy = {} Newton-meters\nMx = {} Newton-meters^2\nMy = {} Newton-meters^2\nMxy = {} Newton-meters^2\n".format(Nx_,Ny_,Nxy_,Mx_,My_,Mxy_))
    return Nx,Ny,Nxy,Mx,My,Mxy

def inv_laminate_constitutive_relations(Nx,Ny,Nxy,Mx,My,Mxy,A_prime,B_prime,C_prime,D_prime):
    large_matrix = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],dtype='float64')
    large_matrix[0:3,0:3] += A_prime
    large_matrix[0:3,3:6] += B_prime
    large_matrix[3:6,0:3] += C_prime
    large_matrix[3:6,3:6] += D_prime
    print("resultant matrix = \n{}".format(large_matrix))
    vector = np.array([Nx,Ny,Nxy,Mx,My,Mxy])
    strain_x0,strain_y0,strain_xy0,kx,ky,kxy = (10**-3)*np.matmul(large_matrix,vector)
    print("strain_x0 = {} millimeters\nstrain_y0 = {} millimeters\nstrain_xy0 = {} millimeters\nkx = {} 1/millimeters\nky = {} 1/millimeters\nkxy = {} 1/millimeters\n".format(strain_x0,strain_y0,strain_xy0,kx,ky,kxy))
    return strain_x0,strain_y0,strain_xy0,kx,ky,kxy

# def inv_midplane_strains(strain_x0,strain_y0,strain_xy0,A_prime,B_prime):
#     large_matrix = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
#     large_matrix[0:3,0:3] += A_prime
#     large_matrix[3:6,0:3] += B_prime
#     print("resultant matrix = \n{}".format(large_matrix))
#     vector = np.array([Nx,Ny,Nxy,Mx,My,Mxy])
#     strain_x0,strain_y0,strain_xy0 = (10**-3)*np.matmul(large_matrix,vector)
#     print("Midplane Strains: strain_x0 = {} millimeters\nstrain_y0 = {} millimeters\nstrain_xy0 = {} millimeters\n".format(strain_x0,strain_y0,strain_xy0))
#     return strain_x0,strain_y0,strain_xy0

def inv_ply_strain_stress(strain_x0,strain_y0,strain_xy0,kx,ky,kxy,layer_Qbars,midpoint_layer_heights):
    stresses = []
    for i in range(len(midpoint_layer_heights)-1):
        print("Layer {}:\n".format(i+1))
        ply_strains = np.array([strain_x0,strain_y0,strain_xy0]) + (((midpoint_layer_heights[i]+midpoint_layer_heights[i+1])/2)*np.matmul(layer_Qbars[i],np.array([kx,ky,kxy])))
        print("x-strain = {} GPa, y-strain = {} GPa, xy-shear strain = {} GPa".format(ply_strains[0],ply_strains[1],ply_strains[2]))
        ply_stresses = (10**3)*np.matmul(layer_Qbars[i],ply_strains)
        stresses.append(ply_stresses)
        print("x-stress = {} 10^-3 GPa, y-stress = {} 10^-3 GPa, xy-shear stress = {} 10^-3 GPa".format(ply_stresses[0],ply_stresses[1],ply_stresses[2]))
    return stresses

def properties_calc1(theta,EL,ET,GLT,vLT):
    Ex_reci = ((np.cos(theta)**4)/EL) + ((np.sin(theta)**4)/ET) + (0.25*((1/GLT)-((2*vLT)/EL))*(np.sin(2*theta)**2))
    Ex = 1/Ex_reci
    Ey_reci = ((np.sin(theta)**4)/EL) + ((np.cos(theta)**4)/ET) + (0.25*((1/GLT)-((2*vLT)/EL))*(np.sin(2*theta)**2))
    Ey = 1/Ey_reci
    vXY = Ex * ( (vLT/EL) - (0.25*((1/EL) + ((2*vLT)/EL) + (1/ET) - (1/GLT))*(np.sin(2*theta)**2)) )
    m_x = np.sin(2*theta) * (vLT + (EL/ET) - (EL/(2*GLT)) - ((np.cos(theta)**2)*(1 + (2*vLT) + (EL/ET) - (EL/GLT))))
    m_y = np.sin(2*theta) * (vLT + (EL/ET) - (EL/(2*GLT)) - ((np.sin(theta)**2)*(1 + (2*vLT) + (EL/ET) - (EL/GLT))))
    GXY_reci = (1/EL) + ((2*vLT)/EL) + (1/ET) - (((1/EL) + ((2*vLT)/EL) + (1/ET) - (1/GLT))*(np.cos(2*theta)**2))
    GXY = 1/GXY_reci
    # print("Ex_reci: {}\nEx: {}\nEy_reci: {}\nEy: {}\nvXY: {}\nm_x: {}\nm_y: {}\nGXY_reci: {}\nGXY: {}".format(Ex_reci,Ex,Ey_reci,Ey,vXY,m_x,m_y,GXY_reci,GXY))
    return Ex_reci,Ex,Ey_reci,Ey,vXY,m_x,m_y,GXY_reci,GXY

def comp_stiff_matrix(EL,ET,GLT,vLT):
    S11 = 1/EL
    S22 = 1/ET
    S12 = -vLT/EL
    vTL = (ET*vLT)/EL
    S66 = 1/GLT
    Q11 = EL / (1-(vLT*vTL))
    Q22 = ET / (1-(vLT*vTL))
    Q12 = (vLT*ET) / (1-(vLT*vTL))
    Q66 = GLT
    # check1 = abs(vLT) < ((EL/ET)**0.5)
    # check2 = abs(vTL) < ((ET/EL)**0.5)
    # print(check1,check2)
    # return(S11,S22,S12,vTL,S66,Q11,Q22,Q12,Q66)
    Q = np.array([[Q11,Q12,0],[Q12,Q22,0],[0,0,Q66]])
    S = np.array([[S11,S12,0],[S12,S22,0],[0,0,S66]])
    return Q,S

def transform_T1(theta): # radians
    T1 = np.array([[np.cos(theta)**2,np.sin(theta)**2,2*np.sin(theta)*np.cos(theta)],
                   [np.sin(theta)**2,np.cos(theta)**2,-2*np.sin(theta)*np.cos(theta)],
                   [-np.sin(theta)*np.cos(theta),np.sin(theta)*np.cos(theta),(np.cos(theta)**2)-(np.sin(theta)**2)]])
    return T1
def transform_T2(theta): # radians
    T2 = np.array([[np.cos(theta)**2,np.sin(theta)**2,np.sin(theta)*np.cos(theta)],
                   [np.sin(theta)**2,np.cos(theta)**2,-np.sin(theta)*np.cos(theta)],
                   [-2*np.sin(theta)*np.cos(theta),2*np.sin(theta)*np.cos(theta),(np.cos(theta)**2)-(np.sin(theta)**2)]])
    return T2

def matrix_zeroing(matrix):
    for i in range(3):
        for j in range(3):
            if abs(matrix[i,j]) < (10**-12):
                matrix[i,j] = 0
    return matrix

def vector_zeroing(vector):
    for i in range(3):
        if abs(vector[i]) < (10**-12):
            vector[i] = 0
    return vector

def Nx_epsx(A_matrix): # rounding throughout the process leads to error in example
    # A_matrix = np.array([[43.375,13.125,0],[13.125,43.375,0],[0,0,15.305]])
    strain_x0,strain_y0,strain_xy0 = np.matmul(np.linalg.inv(A_matrix),np.array([1,0,0]))
    Nx = 1 / strain_x0
    print("Nx = {} * strain_x0".format(Nx))
    return Nx,strain_x0,strain_y0,strain_xy0

def LT_strainx0(strain_x0,strain_y0,strain_xy0,Qbars,T1s):
    # matrix = np.matmul(T1,Qbar)
    ey0_ex0,exy0_ex0 = [strain_y0 / strain_x0,strain_xy0 / strain_x0]
    # ey0_ex0,exy0_ex0 = [-0.3,0]
    LT_vects = []
    for i in range(len(Qbars)):
        vect = np.matmul(np.matmul(T1s[i],Qbars[i]),np.array([1,ey0_ex0,exy0_ex0]))
        vect = vector_zeroing(vect)
        print("LT-stresses(ex0) Layer {} =\n{}".format(i+1,vect))
        LT_vects.append(vect)
    return LT_vects

def failure_test(length,strain_x0,sigLU,sigTU,sigLU_,sigTU_,shearLTU,Qbars,LT_stress_vectors,LT_strain_prior=[]):
    LD,lowest = [{},[]]
    if LT_strain_prior == []: # if this is the first iteration
        for i in range(len(LT_stress_vectors)):
            LD[i+1] = {"Qbar":Qbars[i],"LTstress":LT_stress_vectors[i],"LTstress_":[]}
            if LD[i+1]["LTstress"][0] > 0:
                if LD[i+1]["LTstress"][0] == 0:
                    val = 10**10
                else:
                    val = abs(sigLU*(10**-3) / LD[i+1]["LTstress"][0])
                LD[i+1]["LTstress_"].append(val)
            else:
                if LD[i+1]["LTstress"][0] == 0:
                    val = 10**10
                else:
                    val = abs(sigLU_*(10**-3) / LD[i+1]["LTstress"][0])
                LD[i+1]["LTstress_"].append(val)
            if LD[i+1]["LTstress"][1] > 0:
                if LD[i+1]["LTstress"][1] == 0:
                    val = 10**10
                else:
                    val = abs(sigTU*(10**-3) / LD[i+1]["LTstress"][1])
                LD[i+1]["LTstress_"].append(val)
            else:
                if LD[i+1]["LTstress"][1] == 0:
                    val = 10**10
                else:
                    val = abs(sigTU_*(10**-3) / LD[i+1]["LTstress"][1])
                LD[i+1]["LTstress_"].append(val)
            if LD[i+1]["LTstress"][2] == 0:
                val = 10**10
            else:
                val = abs(shearLTU*(10**-3) / LD[i+1]["LTstress"][2])
            LD[i+1]["LTstress_"].append(val)
            low = min(LD[i+1]["LTstress_"])
            low_ind = LD[i+1]["LTstress_"].index(low)
            lowest.append((low_ind,low))
    else:
        for i in range(len(LT_stress_vectors)):
            LD[i+1] = {"Qbar":Qbars[i],"LTstress":LT_stress_vectors[i],"LTstress_":[]}
            if LD[i+1]["LTstress"][0] > 0:
                if LD[i+1]["LTstress"][0] == 0:
                    val = 10**10
                else:
                    val = abs((sigLU*(10**-3)-LT_strain_prior[i][0]) / LD[i+1]["LTstress"][0])
                LD[i+1]["LTstress_"].append(val)
            else:
                if LD[i+1]["LTstress"][0] == 0:
                    val = 10**10
                else:
                    val = abs((sigLU_*(10**-3)-LT_strain_prior[i][0]) / LD[i+1]["LTstress"][0])
                LD[i+1]["LTstress_"].append(val)
            if LD[i+1]["LTstress"][1] > 0:
                if LD[i+1]["LTstress"][1] == 0:
                    val = 10**10
                else:
                    val = abs((sigTU*(10**-3)-LT_strain_prior[i][1]) / LD[i+1]["LTstress"][1])
                LD[i+1]["LTstress_"].append(val)
            else:
                if LD[i+1]["LTstress"][1] == 0:
                    val = 10**10
                else:
                    val = abs((sigTU_*(10**-3)-LT_strain_prior[i][1]) / LD[i+1]["LTstress"][1])
                LD[i+1]["LTstress_"].append(val)
            if LD[i+1]["LTstress"][2] == 0:
                val = 10**10
            else:
                val = abs((shearLTU*(10**-3)-LT_strain_prior[i][2]) / LD[i+1]["LTstress"][2])
            LD[i+1]["LTstress_"].append(val)
            low = min(LD[i+1]["LTstress_"])
            low_ind = LD[i+1]["LTstress_"].index(low)
            lowest.append((low_ind,low))
    actual_lowest = [i[1] for i in lowest]
    low = min(actual_lowest)
    low_ind = actual_lowest.index(low) # ply where failure occurs
    indexes = []
    for i in range(len(actual_lowest)):
        if actual_lowest[i] == low:
            indexes.append(i)
    print("Failed Layers: {}".format([i+1 for i in indexes]))
    elongation = low * length
    Nval = low / strain_x0 # strain_x0 at failure * conversion equation to Nx = Nx 
    LT_strain_prior = [LD[i+1]["LTstress"]*low for i in range(len(LT_stress_vectors))]
    return Nval,elongation,indexes,LT_strain_prior

def max_stress_failure_testing(sigLU,sigTU,sigLU_,sigTU_,shearLTU,length,layers,layer_heights,layer_T1s,layer_Qbars,midpoint_layer_heights):
    elongations,Nvals,inc = [[],[],0]
    while len(layers) != 0:
        print("\nPly failure {}:".format(inc+1))
        A_matrix = A_matrix_assembly(midpoint_layer_heights,layer_Qbars,layers)
        # B_matrix = B_matrix_assembly(midpoint_layer_heights,layer_Qbars,layers)
        # D_matrix = D_matrix_assembly(midpoint_layer_heights,layer_Qbars,layers)
        Nx,strain_x0,strain_y0,strain_xy0 = Nx_epsx(A_matrix)
        LT_stress_vectors = LT_strainx0(strain_x0,strain_y0,strain_xy0,layer_Qbars,layer_T1s)
        if inc == 0:
            Nval,elongation,indexes,LT_strain_prior = failure_test(length,strain_x0,sigLU,sigTU,sigLU_,sigTU_,shearLTU,layer_Qbars,LT_stress_vectors)
        else:
            Nval,elongation,indexes,LT_strain_prior = failure_test(length,strain_x0,sigLU,sigTU,sigLU_,sigTU_,shearLTU,layer_Qbars,LT_stress_vectors,LT_strain_prior)
            Nval,elongation = [Nvals[-1]+Nval,elongations[-1]+elongation]
        for i,j in enumerate(indexes):
            removed_elements = [layers.pop(j-i),layer_heights.pop(j-i),layer_T1s.pop(j-i),layer_Qbars.pop(j-i),LT_strain_prior.pop(j-i)]
        elongations.append(elongation)
        Nvals.append(Nval)
        inc += 1
    print(elongations)
    print(Nvals)
    return elongations,Nvals

# #### HW6 Example
# print("HW6 example:")
# EL,ET,GLT,vLT = [40,10,4,0.285]
# sigLU,sigTU,sigLU_,sigTU_,shearLTU = [1050,20,650,140,65]
# length = 200 # mm's
# Q,S = comp_stiff_matrix(EL,ET,GLT,vLT)
# print("Q-matrix =\n{}\nS-matrix =\n{}".format(Q,S))
# T1_0,T2_0 = transform_T1(0*np.pi/180),transform_T2(0*np.pi/180)
# T1_90,T2_90 = transform_T1(90*np.pi/180),transform_T2(90*np.pi/180)
# T1_45,T2_45 = transform_T1(45*np.pi/180),transform_T2(45*np.pi/180)
# T1__45,T2__45 = transform_T1(-45*np.pi/180),transform_T2(-45*np.pi/180)
# Qbar_0 = np.matmul(np.linalg.inv(T1_0),np.matmul(Q,T2_0))
# Qbar_90 = np.matmul(np.linalg.inv(T1_90),np.matmul(Q,T2_90))
# Qbar_45 = np.matmul(np.linalg.inv(T1_45),np.matmul(Q,T2_45))
# Qbar__45 = np.matmul(np.linalg.inv(T1__45),np.matmul(Q,T2__45))
# Qbar_0 = matrix_zeroing(Qbar_0)
# Qbar_90 = matrix_zeroing(Qbar_90)
# Qbar_45 = matrix_zeroing(Qbar_45)
# Qbar__45 = matrix_zeroing(Qbar__45)
# # Sbar_0 = np.matmul(np.linalg.inv(T2_0),np.matmul(S,T1_0))
# # Sbar_90 = np.matmul(np.linalg.inv(T2_90),np.matmul(S,T1_90))
# # Sbar_45 = np.matmul(np.linalg.inv(T2_45),np.matmul(S,T1_45))
# # Sbar__45 = np.matmul(np.linalg.inv(T2__45),np.matmul(S,T1__45))
# # Sbar_0 = matrix_zeroing(Sbar_0)
# # Sbar_90 = matrix_zeroing(Sbar_90)
# # Sbar_45 = matrix_zeroing(Sbar_45)
# # Sbar__45 = matrix_zeroing(Sbar__45)
# Q_1,Q_2,Q_3,Q_4,Q_5,Q_6,Q_7,Q_8 = [Qbar_0,Qbar_45,Qbar__45,Qbar_90,Qbar_90,Qbar__45,Qbar_45,Qbar_0]
# # S_1,S_2,S_3,S_4,S_5,S_6,S_7,S_8 = [Sbar_0,Sbar_45,Sbar__45,Sbar_90,Sbar_90,Sbar__45,Sbar_45,Sbar_0]
# layers = [1,2,3,4,5,6,7,8]
# layer_heights = [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25] # mm's
# layer_T1s = [T1_0,T1_45,T1__45,T1_90,T1_90,T1__45,T1_45,T1_0]
# layer_Qbars = [Q_1,Q_2,Q_3,Q_4,Q_5,Q_6,Q_7,Q_8]
# # layer_Sbars = [S_1,S_2,S_3,S_4,S_5,S_6,S_7,S_8]
# for i in range(len(layer_Qbars)):
#     print("Layer {} Q_bar matrix = \n{} GPa".format(i+1,layer_Qbars[i]))
# sum_layer_heights = sum(layer_heights)
# half_height = sum_layer_heights/2
# midpoint_layer_heights = [-half_height]
# for i in range(len(layer_heights)):
#     midpoint_layer_heights.append(midpoint_layer_heights[i]+layer_heights[i])
# print(midpoint_layer_heights)
# elongations,Nvals = max_stress_failure_testing(sigLU,sigTU,sigLU_,sigTU_,shearLTU,length,layers,layer_heights,layer_T1s,layer_Qbars,midpoint_layer_heights)
# elongations,Nvals = [[0]+elongations,[0]+Nvals]
# plt.plot(elongations,Nvals,color="blue")
# for i in range(len(elongations)):
#     plt.plot(elongations[i],Nvals[i],color="black",marker="o",label="{} kN".format(round(Nvals[i],3)))
# plt.xlabel("millimeters (mm)")
# plt.ylabel("kilo-Newtons (kN)")
# plt.title("[0/45/-45/90]s 0.25mm laminate")
# plt.grid()
# plt.legend()
# plt.show()
# print()

#### HW6 Problem
print("HW6 problem:")
EL,ET,GLT,vLT = [40,10,4,0.285]
sigLU,sigTU,sigLU_,sigTU_,shearLTU = [1050,20,650,140,65]
length = 300 # mm's
Q,S = comp_stiff_matrix(EL,ET,GLT,vLT)
print("Q-matrix =\n{}\nS-matrix =\n{}".format(Q,S))
T1_0,T2_0 = transform_T1(0*np.pi/180),transform_T2(0*np.pi/180)
T1_90,T2_90 = transform_T1(90*np.pi/180),transform_T2(90*np.pi/180)
T1_30,T2_30 = transform_T1(30*np.pi/180),transform_T2(30*np.pi/180)
T1__30,T2__30 = transform_T1(-30*np.pi/180),transform_T2(-30*np.pi/180)
Qbar_0 = np.matmul(np.linalg.inv(T1_0),np.matmul(Q,T2_0))
Qbar_90 = np.matmul(np.linalg.inv(T1_90),np.matmul(Q,T2_90))
Qbar_30 = np.matmul(np.linalg.inv(T1_30),np.matmul(Q,T2_30))
Qbar__30 = np.matmul(np.linalg.inv(T1__30),np.matmul(Q,T2__30))
Qbar_0 = matrix_zeroing(Qbar_0)
Qbar_90 = matrix_zeroing(Qbar_90)
Qbar_30 = matrix_zeroing(Qbar_30)
Qbar__30 = matrix_zeroing(Qbar__30)
# Sbar_0 = np.matmul(np.linalg.inv(T2_0),np.matmul(S,T1_0))
# Sbar_90 = np.matmul(np.linalg.inv(T2_90),np.matmul(S,T1_90))
# Sbar_45 = np.matmul(np.linalg.inv(T2_45),np.matmul(S,T1_45))
# Sbar__45 = np.matmul(np.linalg.inv(T2__45),np.matmul(S,T1__45))
# Sbar_0 = matrix_zeroing(Sbar_0)
# Sbar_90 = matrix_zeroing(Sbar_90)
# Sbar_45 = matrix_zeroing(Sbar_45)
# Sbar__45 = matrix_zeroing(Sbar__45)
Q_1,Q_2,Q_3,Q_4,Q_5,Q_6,Q_7,Q_8 = [Qbar_0,Qbar_30,Qbar_90,Qbar__30,Qbar__30,Qbar_90,Qbar_30,Qbar_0]
# S_1,S_2,S_3,S_4,S_5,S_6,S_7,S_8 = [Sbar_0,Sbar_45,Sbar__45,Sbar_90,Sbar_90,Sbar__45,Sbar_45,Sbar_0]
layers = [1,2,3,4,5,6,7,8]
layer_heights = [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25] # mm's
layer_T1s = [T1_0,T1_30,T1_90,T1__30,T1__30,T1_90,T1_30,T1_0]
layer_Qbars = [Q_1,Q_2,Q_3,Q_4,Q_5,Q_6,Q_7,Q_8]
# layer_Sbars = [S_1,S_2,S_3,S_4,S_5,S_6,S_7,S_8]
for i in range(len(layer_Qbars)):
    print("Layer {} Q_bar matrix = \n{} GPa".format(i+1,layer_Qbars[i]))
sum_layer_heights = sum(layer_heights)
half_height = sum_layer_heights/2
midpoint_layer_heights = [-half_height]
for i in range(len(layer_heights)):
    midpoint_layer_heights.append(midpoint_layer_heights[i]+layer_heights[i])
print(midpoint_layer_heights)
elongations,Nvals = max_stress_failure_testing(sigLU,sigTU,sigLU_,sigTU_,shearLTU,length,layers,layer_heights,layer_T1s,layer_Qbars,midpoint_layer_heights)
elongations,Nvals = [[0]+elongations,[0]+Nvals]
plt.plot(elongations,Nvals,color="blue")
for i in range(len(elongations)):
    plt.plot(elongations[i],Nvals[i],color="black",marker="o",label="{} kN".format(round(Nvals[i],3)))
plt.xlabel("millimeters (mm)")
plt.ylabel("kilo-Newtons (kN)")
plt.title("[0/30/90/-30]s 0.25mm laminate")
plt.grid()
plt.legend()
plt.show()
print()