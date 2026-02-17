# Camden Hill
# A1D Piecewise Thermal FEA
from sympy import log
import numpy as np
import matplotlib.pyplot as plt

def case12_analytic_setup(x0,xmid,xL,alpha1,alpha2,k1,k2,A1,A2,T0,T_0,TL,Ta,BCs,Disc):
    if len(Disc) == 2:
        if BCs == (0,0): # case 1
            matrix = np.array([
            [np.exp(alpha1*x0),np.exp(-alpha1*x0),0,0],
            [0,0,np.exp(alpha2*xL),np.exp(-alpha2*xL)],
            [k1*A1*alpha1*np.exp(alpha1*xmid),-k1*A1*alpha1*np.exp(-alpha1*xmid),-k2*A2*alpha2*np.exp(alpha2*xmid),k2*A2*alpha2*np.exp(-alpha2*xmid)],
            [np.exp(alpha1*xmid),np.exp(-alpha1*xmid),-np.exp(alpha2*xmid),-np.exp(-alpha2*xmid)]])
            vector = np.array([T0-Ta,TL-Ta,0,0])
        elif BCs == (1,0):# case 2
            matrix = np.array([
            [k1*A1*alpha1*np.exp(alpha1*x0),-k1*A1*alpha1*np.exp(-alpha1*x0),0,0],
            [0,0,np.exp(alpha2*xL),np.exp(-alpha2*xL)],
            [k1*A1*alpha1*np.exp(alpha1*xmid),-k1*A1*alpha1*np.exp(-alpha1*xmid),-k2*A2*alpha2*np.exp(alpha2*xmid),k2*A2*alpha2*np.exp(-alpha2*xmid)],
            [np.exp(alpha1*xmid),np.exp(-alpha1*xmid),-np.exp(alpha2*xmid),-np.exp(-alpha2*xmid)]])
            vector = np.array([T_0,TL-Ta,0,0])
        c1,c2,c3,c4 = np.linalg.solve(matrix,vector)
        def case12_Tx_analytic(x):
            if x < xmid:
                return (c1*np.exp(alpha1*x)) + (c2*np.exp(-alpha1*x)) + Ta
            else:
                return (c3*np.exp(alpha2*x)) + (c4*np.exp(-alpha2*x)) + Ta
        def case12_Qdx_analytic(x):
            if x < xmid:
                return -k1*A1*((c1*alpha1*np.exp(alpha1*x)) - (c2*alpha1*np.exp(-alpha1*x)))
            else:
                return -k2*A2*((c3*alpha2*np.exp(alpha2*x)) - (c4*alpha2*np.exp(-alpha2*x)))
        ########### Analytical solutions are identical between my code and the example
        return case12_Tx_analytic, case12_Qdx_analytic
    elif len(Disc) == 1:
        if BCs == (0,0):
            matrix = np.array([[np.exp(alpha1*x0),np.exp(-alpha1*x0)],[np.exp(alpha1*xL),np.exp(-alpha1*xL)]])
            vector = np.array([T0-Ta,TL-Ta])
            c1,c2 = np.linalg.solve(matrix,vector)
            # def case12_Tx_analytic(x):
            #     return (((TL-Ta)/np.sinh(alpha1*L))*np.sinh(alpha1*x))+Ta
            # def case12_Qdx_analytic(x):
            #     return (-k1*A1)*((((TL-Ta)/np.sinh(alpha1*L))*alpha1*np.cosh(alpha1*x)))
        elif BCs == (1,0):
            matrix = np.array([[k1*A1*alpha1*np.exp(alpha1*x0),-k1*A1*alpha1*np.exp(-alpha1*x0)],[np.exp(alpha1*xL),np.exp(-alpha1*xL)]])
            vector = np.array([T_0,TL-Ta])
            c1,c2 = np.linalg.solve(matrix,vector)
            # def case12_Tx_analytic(x):
            #     return (((TL-Ta)/np.cosh(alpha1*L))*np.cosh(alpha1*x))+Ta
            # def case12_Qdx_analytic(x):
            #     return (-k1*A1)*((((TL-Ta)/np.cosh(alpha1*L))*alpha1*np.sinh(alpha1*x)))
        def case12_Tx_analytic(x):
            return (c1*np.exp(alpha1*x)) + (c2*np.exp(-alpha1*x)) + Ta
        def case12_Qdx_analytic(x):
            return -k1*A1*((c1*alpha1*np.exp(alpha1*x)) - (c2*alpha1*np.exp(-alpha1*x)))
        return case12_Tx_analytic, case12_Qdx_analytic
    else:
        print("Not programmed")

def nonstandard_mesh(Disc,L,i):
    if Disc != [0]:
        # # step = L/i
        # bins = [2 for j in range(len(Disc))]
        # # ind,pos = [0,step]
        # Disc_ = Disc+[L]
        # # for j in range(1,i):
        # #     pos = step * j
        # #     if pos < Disc_[ind+1]:
        # #         bins[ind] += 1
        # #     else:
        # #         ind += 1
        # #     print(bins)
        # points_sum = sum(bins)
        # while points_sum > i:
        #     ind = bins.index(max(bins))
        #     if bins[ind] <= 1:
        #         print("Error: p = {}, BCs = {}, Disc. = {}, i = {}".format(p,BCs,Disc_,i))
        #     else:
        #         bins[ind] -= 1
        #         points_sum -= 1
        #     # print(bins)
        # ranges = [abs((Disc_[j+1]-Disc_[j]))/bins[j] for j in range(len(Disc_)-1)]
        # while points_sum < i:
        #     ind = ranges.index(max(ranges))
        #     bins[ind] += 1
        #     ranges[ind] = abs((Disc_[ind+1]-Disc_[ind]))/bins[ind]
        #     points_sum += 1
        #     # print(bins)
        # points = [np.array([],dtype='float64')]
        # for j in range(len(Disc_)-1):
        #     space = np.linspace(Disc_[j],Disc_[j+1],bins[j]+1)
        #     if j > 0:
        #         space = space[1:]
        #     points = np.append(points,space)
        # # Disc = Disc[:-1]
        # breaks = [sum(bins[0:j+1]) for j in range(len(Disc_)-1)]
        ########
        # points = np.append(np.linspace(Disc[0],Disc[1],int(i/2)+1)[:-1],np.linspace(Disc[1],L,int(i/2)+1))
        # breaks = [int(i/2),i]
        #######
        fraction = int((Disc[1]/L)*i)
        breaks = [fraction,i-fraction]
        points = np.append(np.linspace(Disc[0],Disc[1],breaks[0]+1)[:-1],np.linspace(Disc[1],L,breaks[1]+1))
        return breaks,points
    else:
        return [[],np.linspace(0,L,i+1)]

#### Problem Coefficients
h = 0.4             # Heat Transfer Coefficient 
R = 0.1             # Cross section radius
Ac = np.pi * R**2   # Cross sectional area
Pc = 2 * np.pi * R      # Cross sectional perimeter
L = 1               # Bar length
As = 2*np.pi*R*L
# k = (h*2)/((a**2)*R)

# Case paramaters
T0 = 0             # T(0)
T_0 = 0           # T0dx(0)
TL = 100           # T(L)
Ta = 0              # Ambient temperature

######## Finite Element Method
points = [("FDM",2),("FDM",4),("FEM",1),("FEM",2)]
alphas = [0.9,2.75,3.25,4.75] # [0.29,0.50,1.35,2.75,4.75,9.15]
# ks = [(h*2)/((a**2)*R) for a in alphas]
sams = [4,8,16,32,64,128] # [2,4,8] [2,4,8,16,32,64,128]
VD = {}
BCs = (0,0)
Disc = [0,(2/np.pi)*L] # [0,(2/np.pi)*L] [0]
Disc_vals = {Disc[0]:1.35,Disc[1]:alphas} # {Disc[0]:1.35} {Disc[0]:1.35,Disc[1]:alphas}

SD = {}
for i in sams: # works only on sample size greater than 4
    SD[i] = nonstandard_mesh(Disc,L,i)
# print(SD)
for p in points: # works only on sample size greater than 4
    VD[p] = {}
    for a in alphas:
        VD[p][a] = {}
        if p == ("FDM",2):
            if len(Disc) == 1:
                for i in sams:
                    step,k = [L/i,2+((a**2)*((L/i)**2))]
                    i += 1
                    K,F,Qdx2 = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                    if BCs == (0,0): # case 1
                        K[0,0],K[i-1,i-1] = (1,1)
                    elif BCs == (1,0): # case 2
                        K[0,0:2],K[i-1,i-1] = ((k/2,-1),1)
                    for j in range(2,i):
                        K[j-1,j-2:j+1] = (-1,k,-1)
                    F[i-1] = TL
                    B = np.linalg.solve(K,F)
                    case12_Tx_analytic, case12_Qdx_analytic = case12_analytic_setup(0,0,L,a,0,(h*2)/((a**2)*R),0,Ac,0,T0,T_0,TL,Ta,BCs,Disc)
                    ind_of_interest,point_of_interest = [(i-1)//2,SD[i-1][1][(i-1)//2]] # relatively in the middle
                    TL_an, Qdx_an = [case12_Tx_analytic(L/2),case12_Qdx_analytic(L/2)]
                    T_err = float(log(abs((B[ind_of_interest] - TL_an)/TL_an),10))
                    for j in range(1,i):
                        Qdx2[j-1] = (-(h*2)/((a**2)*R)*Ac)*(((B[j]-B[j-1])/step)+(((a**2)*step*B[j])/2))
                    Qdx_err = float(log(abs((Qdx2[ind_of_interest-1] - Qdx_an)/Qdx_an),10))
                    VD[p][a][i-1] = {"xvals":SD[i-1][1],"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"HeatTran2":Qdx2,"T_error":T_err,"Qdx_err":Qdx_err,"Tx_eq":case12_Tx_analytic,"Qdx_eq":case12_Qdx_analytic}
            elif len(Disc) == 2:
                    k1,k2 = [(h*2)/((Disc_vals[0]**2)*R),(h*2)/((a**2)*R)]
                    for i in sams:
                        i += 1
                        # step1,step2 = [SD[i-1][1][1]-SD[i-1][1][0],SD[i-1][1][i-1]-SD[i-1][1][i-2]]
                        # k1,k2 = [2+((Disc_vals[0]**2)*((step1)**2)),2+((a**2)*((step2)**2))]
                        K,F,Qdx2 = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                        for j in range(1,i-1):
                            step = SD[i-1][1][j]-SD[i-1][1][j-1]
                            if j == SD[i-1][0][0]: # if j == index of discontinuity
                                step2 = SD[i-1][1][j+1]-SD[i-1][1][j]
                                K[j,j-1:j+2] = (-k1*Ac/(step),(k1*Ac/step)+(k2*Ac/step2)+(h*Pc*step/2)+(h*Pc*step2/2),-k2*Ac/(step2))
                            elif j < SD[i-1][0][0]:
                                K[j,j-1:j+2] = (-k1*Ac/(step),(2*k1*Ac/step)+(h*Pc*step),-k1*Ac/(step))
                            elif j > SD[i-1][0][0]:
                                K[j,j-1:j+2] = (-k2*Ac/(step),(2*k2*Ac/step)+(h*Pc*step),-k2*Ac/(step))
                        # for j in range(2,i):
                        #     K[j-1,j-2:j+1] = (-1,k,-1)
                        if BCs == (0,0): # case 1
                            K[0,0:2],K[i-1,i-2:i] = ((1,0),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                        elif BCs == (1,0): # case 2
                            step = SD[i-1][1][1]-SD[i-1][1][0]
                            K[0,0:2],K[i-1,i-2:i] = ((((2*k1*Ac/step)+(h*Pc*step))/2,-k1*Ac/(step)),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                            # K[0,0:2],K[i-1,i-2:i] = ((-k1*Ac/(step),-k1*Ac/(step)),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                        F[i-1] = TL
                        B = np.linalg.solve(K,F)
                        case12_Tx_analytic, case12_Qdx_analytic = case12_analytic_setup(0,Disc[1],L,Disc_vals[0],a,k1,k2,Ac,Ac,T0,T_0,TL,Ta,BCs,Disc)
                        ind_of_interest,point_of_interest = [SD[i-1][0][0],SD[i-1][1][SD[i-1][0][0]]] # relatively in the middle # ind_of_interest,point_of_interest = [(i-1)//2,SD[i-1][1][(i-1)//2]] # relatively in the middle
                        TL_an, Qdx_an = [case12_Tx_analytic(point_of_interest),case12_Qdx_analytic(point_of_interest)]
                        T_err = float(log(abs((B[ind_of_interest] - TL_an)/TL_an),10))
                        for j in range(1,i):
                            step = SD[i-1][1][j]-SD[i-1][1][j-1]
                            if j <= SD[i-1][0][0]:
                                Qdx2[j-1] = (-k1*Ac)*(((B[j]-B[j-1])/step)+(((Disc_vals[0]**2)*step*B[j])/2))
                            elif j > SD[i-1][0][0]:
                                Qdx2[j-1] = (-k2*Ac)*(((B[j]-B[j-1])/step)+(((a**2)*step*B[j])/2))
                        Qdx_err = float(log(abs((Qdx2[ind_of_interest-1] - Qdx_an)/Qdx_an),10))
                        VD[p][a][i-1] = {"xvals":SD[i-1][1],"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"HeatTran2":Qdx2,"T_error":T_err,"Qdx_err":Qdx_err,"Tx_eq":case12_Tx_analytic,"Qdx_eq":case12_Qdx_analytic}
        elif p == ("FDM",4):
            if len(Disc) == 1:
                k = (h*2)/((a**2)*R)
                for i in sams:
                    step = L/i
                    i += 1
                    K,F,Qdx4 = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                    k_matrix = np.array([[(-k*Ac)*(144+(72*(step**2)*(h*Pc/(k*Ac)))+(6*(step**4)*((h*Pc/(k*Ac))**2))) / ((144*step)+(24*(step**3)*(h*Pc/(k*Ac)))),
                                          144*k*Ac/((144*step)+(24*(step**3)*h*Pc/(k*Ac)))],
                                          [144*k*Ac/((144*step)+(24*(step**3)*h*Pc/(k*Ac))),
                                            (-k*Ac)*(144+(72*(step**2)*(h*Pc/(k*Ac)))+(6*(step**4)*((h*Pc/(k*Ac))**2))) / ((144*step)+(24*(step**3)*(h*Pc/(k*Ac))))]])
                    for j in range(1,i):
                        K[j-1:j+1,j-1:j+1] += k_matrix
                    if BCs == (0,0): # case 1
                        K[0,0:2],K[i-1,i-2:i] = ((1,0),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    elif BCs == (1,0): # case 2
                        K[0,0:2],K[i-1,i-2:i] = ((k_matrix[0,0],k_matrix[0,1]),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    F[0],F[i-1] = [T0,TL]
                    B = np.linalg.solve(K,F)
                    case12_Tx_analytic, case12_Qdx_analytic = case12_analytic_setup(0,0,L,a,0,k,0,Ac,0,T0,T_0,TL,Ta,BCs,Disc)
                    ind_of_interest,point_of_interest = [(i-1)//2,SD[i-1][1][(i-1)//2]] # relatively in the middle
                    TL_an, Qdx_an = [case12_Tx_analytic(L/2),case12_Qdx_analytic(L/2)]
                    T_err = float(log(abs((B[ind_of_interest] - TL_an)/TL_an),10))
                    for j in range(1,i):
                        # Qdx2[j-1] = (-k*Ac)*(((B[j]-B[j-1])/step)+(((a**2)*step*B[j])/2))
                        Qdx4[j-1] = (-k*Ac)*(((144*B[j])-(144*B[j-1])+(72*(step**2)*(a**2)*B[j])+(6*(step**4)*(a**4)*B[j]))/((144*step)+(24*(step**3)*(a**2))))
                    Qdx_err = float(log(abs((Qdx4[ind_of_interest-1] - Qdx_an)/Qdx_an),10))
                    VD[p][a][i-1] = {"xvals":SD[i-1][1],"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"HeatTran4":Qdx4,"T_error":T_err,"Qdx_err":Qdx_err,"Tx_eq":case12_Tx_analytic,"Qdx_eq":case12_Qdx_analytic}
            elif len(Disc) == 2:
                k1,k2 = [(h*2)/((Disc_vals[0]**2)*R),(h*2)/((a**2)*R)]
                for i in sams:
                    i += 1
                    K,F,Qdx4 = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                    step1,step2 = [SD[i-1][1][1]-SD[i-1][1][0],SD[i-1][1][i-1]-SD[i-1][1][i-2]]
                    k_matrix1 = np.array([[(-k1*Ac)*(144+(72*(step1**2)*(h*Pc/(k1*Ac)))+(6*(step1**4)*((h*Pc/(k1*Ac))**2))) / ((144*step1)+(24*(step1**3)*(h*Pc/(k1*Ac)))),
                                        144*k1*Ac/((144*step1)+(24*(step1**3)*h*Pc/(k1*Ac)))],
                                        [144*k1*Ac/((144*step1)+(24*(step1**3)*h*Pc/(k1*Ac))),
                                            (-k1*Ac)*(144+(72*(step1**2)*(h*Pc/(k1*Ac)))+(6*(step1**4)*((h*Pc/(k1*Ac))**2))) / ((144*step1)+(24*(step1**3)*(h*Pc/(k1*Ac))))]])
                    k_matrix2 = np.array([[(-k2*Ac)*(144+(72*(step2**2)*(h*Pc/(k2*Ac)))+(6*(step2**4)*((h*Pc/(k2*Ac))**2))) / ((144*step2)+(24*(step2**3)*(h*Pc/(k2*Ac)))),
                                        144*k2*Ac/((144*step2)+(24*(step2**3)*h*Pc/(k2*Ac)))],
                                        [144*k2*Ac/((144*step2)+(24*(step2**3)*h*Pc/(k2*Ac))),
                                            (-k2*Ac)*(144+(72*(step2**2)*(h*Pc/(k2*Ac)))+(6*(step2**4)*((h*Pc/(k2*Ac))**2))) / ((144*step2)+(24*(step2**3)*(h*Pc/(k2*Ac))))]])
                    for j in range(1,i):
                        if j < SD[i-1][0][0]+1: # if j == index of discontinuity
                            K[j-1:j+1,j-1:j+1] += k_matrix1
                        else:
                            K[j-1:j+1,j-1:j+1] += k_matrix2
                    if BCs == (0,0): # case 1
                        K[0,0:2],K[i-1,i-2:i] = ((1,0),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    elif BCs == (1,0): # case 2
                        K[0,0:2],K[i-1,i-2:i] = ((k_matrix1[0,0],k_matrix1[0,1]),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    F[0],F[i-1] = [T0,TL]
                    B = np.linalg.solve(K,F)
                    case12_Tx_analytic, case12_Qdx_analytic = case12_analytic_setup(0,Disc[1],L,Disc_vals[0],a,k1,k2,Ac,Ac,T0,T_0,TL,Ta,BCs,Disc)
                    ind_of_interest,point_of_interest = [SD[i-1][0][0],SD[i-1][1][SD[i-1][0][0]]] # relatively in the middle # ind_of_interest,point_of_interest = [(i-1)//2,SD[i-1][1][(i-1)//2]] # relatively in the middle
                    TL_an, Qdx_an = [case12_Tx_analytic(point_of_interest),case12_Qdx_analytic(point_of_interest)]
                    T_err = float(log(abs((B[ind_of_interest] - TL_an)/TL_an),10))
                    for j in range(1,i):
                        step = SD[i-1][1][j]-SD[i-1][1][j-1]
                        if j < SD[i-1][0][0]+1:
                            Qdx4[j-1] = (-k1*Ac)*(((144*B[j])-(144*B[j-1])+(72*(step**2)*(Disc_vals[0]**2)*B[j])+(6*(step**4)*(Disc_vals[0]**4)*B[j]))/((144*step)+(24*(step**3)*(Disc_vals[0]**2))))
                        elif j > SD[i-1][0][0]:
                            Qdx4[j-1] = (-k2*Ac)*(((144*B[j])-(144*B[j-1])+(72*(step**2)*(a**2)*B[j])+(6*(step**4)*(a**4)*B[j]))/((144*step)+(24*(step**3)*(a**2))))
                    Qdx_err = float(log(abs((Qdx4[ind_of_interest-1] - Qdx_an)/Qdx_an),10))
                    VD[p][a][i-1] = {"xvals":SD[i-1][1],"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"HeatTran4":Qdx4,"T_error":T_err,"Qdx_err":Qdx_err,"Tx_eq":case12_Tx_analytic,"Qdx_eq":case12_Qdx_analytic}
        elif p == ("FEM",1):
            if len(Disc) == 1:
                k = (h*2)/((a**2)*R)
                for i in sams:
                    step = L/i
                    i += 1
                    K,F,Qdx2 = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                    k_11,k_12 = [(1/step)+(((a**2)*step)/3),-(1/step)+(((a**2)*step)/6)]
                    for j in range(1,i):
                        K[j-1:j+1,j-1:j+1] += np.array([[k_11,k_12],[k_12,k_11]])
                    if BCs == (0,0): # case 1
                        K[0,0:2],K[i-1,i-2:i] = ((1,0),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    elif BCs == (1,0): # case 2
                        K[0,0:2],K[i-1,i-2:i] = ((k_11,k_12),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    F[0],F[i-1] = [T0,TL]
                    B = np.linalg.solve(K,F)
                    case12_Tx_analytic, case12_Qdx_analytic = case12_analytic_setup(0,0,L,a,0,k,0,Ac,0,T0,T_0,TL,Ta,BCs,Disc)
                    ind_of_interest,point_of_interest = [(i-1)//2,SD[i-1][1][(i-1)//2]] # relatively in the middle
                    TL_an, Qdx_an = [case12_Tx_analytic(L/2),case12_Qdx_analytic(L/2)]
                    T_err = float(log(abs((B[ind_of_interest] - TL_an)/TL_an),10))
                    for j in range(1,i):
                        Qdx2[j-1] = (-k*Ac)*(((B[j]-B[j-1])/step)+(((a**2)*step*B[j])/2))
                    Qdx_err = float(log(abs((Qdx2[ind_of_interest-1] - Qdx_an)/Qdx_an),10))
                    VD[p][a][i-1] = {"xvals":SD[i-1][1],"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"HeatTran2":Qdx2,"T_error":T_err,"Qdx_err":Qdx_err,"Tx_eq":case12_Tx_analytic,"Qdx_eq":case12_Qdx_analytic}
            elif len(Disc) == 2:
                k1,k2 = [(h*2)/((Disc_vals[0]**2)*R),(h*2)/((a**2)*R)]
                for i in sams:
                    i += 1
                    # k1,k2 = [2+((Disc_vals[0]**2)*((SD[i-1][1][1]-SD[i-1][1][0])**2)),2+((a**2)*((SD[i-1][1][i-1]-SD[i-1][1][i-2])**2))]
                    K,F,Qdx2 = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                    if BCs == (0,0): # case 1
                        K[0,0:2],K[i-1,i-2:i] = ((1,0),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    elif BCs == (1,0): # case 2
                        step = SD[i-1][1][1]-SD[i-1][1][0]
                        K[0,0:2],K[i-1,i-2:i] = (((1/step)+(((Disc_vals[0]**2)*step)/3),-(1/step)+(((Disc_vals[0]**2)*step)/6)),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    for j in range(1,i-1):
                        step = SD[i-1][1][j]-SD[i-1][1][j-1]
                        if j == SD[i-1][0][0]: # if j == index of discontinuity
                            step2 = SD[i-1][1][j+1]-SD[i-1][1][j]
                            K[j,j-1:j+2] = (((-k1*Ac/step)+(h*Pc*step/6)),((k1*Ac/step)+(k2*Ac/step2)+(h*Pc*step/3)+(h*Pc*step2/3)),((-k2*Ac/step2)+(h*Pc*step2/6)))
                        elif j < SD[i-1][0][0]:
                            K[j,j-1:j+2] = (-1,-(2*((k1*Ac/step)+(2*h*Pc*step/6)))/((-k1*Ac/step)+(h*Pc*step/6)),-1)
                        elif j > SD[i-1][0][0]:
                            K[j,j-1:j+2] = (-1,-(2*((k2*Ac/step)+(2*h*Pc*step/6)))/((-k2*Ac/step)+(h*Pc*step/6)),-1)
                    F[0],F[i-1] = [T0,TL]
                    B = np.linalg.solve(K,F)
                    case12_Tx_analytic, case12_Qdx_analytic = case12_analytic_setup(0,Disc[1],L,Disc_vals[0],a,k1,k2,Ac,Ac,T0,T_0,TL,Ta,BCs,Disc)
                    ind_of_interest,point_of_interest = [SD[i-1][0][0],SD[i-1][1][SD[i-1][0][0]]] # relatively in the middle # ind_of_interest,point_of_interest = [(i-1)//2,SD[i-1][1][(i-1)//2]] # relatively in the middle
                    TL_an, Qdx_an = [case12_Tx_analytic(point_of_interest),case12_Qdx_analytic(point_of_interest)]
                    T_err = float(log(abs((B[ind_of_interest] - TL_an)/TL_an),10))
                    for j in range(1,i):
                        step = SD[i-1][1][j]-SD[i-1][1][j-1]
                        if j <= SD[i-1][0][0]:
                            Qdx2[j-1] = (-k1*Ac)*(((B[j]-B[j-1])/step)+(((Disc_vals[0]**2)*step*B[j])/2))
                        elif j > SD[i-1][0][0]:
                            Qdx2[j-1] = (-k2*Ac)*(((B[j]-B[j-1])/step)+(((a**2)*step*B[j])/2))
                    Qdx_err = float(log(abs((Qdx2[ind_of_interest-1] - Qdx_an)/Qdx_an),10))
                    VD[p][a][i-1] = {"xvals":SD[i-1][1],"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"HeatTran2":Qdx2,"T_error":T_err,"Qdx_err":Qdx_err,"Tx_eq":case12_Tx_analytic,"Qdx_eq":case12_Qdx_analytic}
        elif p == ("FEM",2):
            if len(Disc) == 1:
                k = (h*2)/((a**2)*R)
                for i in sams:
                    step = L/i
                    i += 1
                    K,F,Qdx4 = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                    k_matrix1 = (k*Ac)*np.array([[1/step,-1/step,0],[-1/step,1/step,0],[0,0,1/(3*step)]])
                    k_matrix2 = (h*Pc)*np.array([[step/3,step/6,step/12],[step/6,step/3,step/12],[step/12,step/12,step/30]])
                    k_matrix = k_matrix1 + k_matrix2
                    B,C = [np.array([[k_matrix[0,2],0],[k_matrix[1,2],0]]),np.array([k_matrix[2,0:2],[0,0]])]
                    k_reduced = k_matrix[0:2,0:2] - np.matmul(B,(1/k_matrix[2,2])*C)
                    # print(k_reduced)
                    # k_reduced = np.array([[k_matrix[0,0]-(k_matrix[0,2]*k_matrix[2,0]/k_matrix[2,2]),k_matrix[0,1]-(k_matrix[0,2]*k_matrix[2,1]/k_matrix[2,2])],
                    #                       [k_matrix[1,0]-(k_matrix[1,2]*k_matrix[2,0]/k_matrix[2,2]),k_matrix[1,1]-(k_matrix[1,2]*k_matrix[2,1]/k_matrix[2,2])]])
                    # print(k_reduced)
                    for j in range(1,i):
                        K[j-1:j+1,j-1:j+1] += k_reduced
                    if BCs == (0,0): # case 1
                        K[0,0:2],K[i-1,i-2:i] = ((1,0),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    elif BCs == (1,0): # case 2
                        K[0,0:2],K[i-1,i-2:i] = ((k_reduced[0,0],k_reduced[0,1]),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    F[0],F[i-1] = [T0,TL]
                    B = np.linalg.solve(K,F)
                    case12_Tx_analytic, case12_Qdx_analytic = case12_analytic_setup(0,0,L,a,0,k,0,Ac,0,T0,T_0,TL,Ta,BCs,Disc)
                    ind_of_interest,point_of_interest = [(i-1)//2,SD[i-1][1][(i-1)//2]] # relatively in the middle
                    TL_an, Qdx_an = [case12_Tx_analytic(L/2),case12_Qdx_analytic(L/2)]
                    T_err = float(log(abs((B[ind_of_interest] - TL_an)/TL_an),10))
                    for j in range(1,i):
                        # Qdx2[j-1] = (-k*Ac)*(((B[j]-B[j-1])/step)+(((a**2)*step*B[j])/2))
                        Qdx4[j-1] = (-k*Ac)*(((144*B[j])-(144*B[j-1])+(72*(step**2)*(a**2)*B[j])+(6*(step**4)*(a**4)*B[j]))/((144*step)+(24*(step**3)*(a**2))))
                    Qdx_err = float(log(abs((Qdx4[ind_of_interest-1] - Qdx_an)/Qdx_an),10))
                    VD[p][a][i-1] = {"xvals":SD[i-1][1],"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"HeatTran4":Qdx4,"T_error":T_err,"Qdx_err":Qdx_err,"Tx_eq":case12_Tx_analytic,"Qdx_eq":case12_Qdx_analytic}
            elif len(Disc) == 2:
                k1,k2 = [(h*2)/((Disc_vals[0]**2)*R),(h*2)/((a**2)*R)]
                for i in sams:
                    i += 1
                    K,F,Qdx4 = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                    step1,step2 = [SD[i-1][1][1]-SD[i-1][1][0],SD[i-1][1][i-1]-SD[i-1][1][i-2]]
                    k_matrix1 = (k1*Ac)*np.array([[1/step1,-1/step1,0],[-1/step1,1/step1,0],[0,0,1/(3*step1)]])
                    k_matrix2 = (h*Pc)*np.array([[step1/3,step1/6,step1/12],[step1/6,step1/3,step1/12],[step1/12,step1/12,step1/30]])
                    k_matrix = k_matrix1 + k_matrix2
                    B,C = [np.array([[k_matrix[0,2],0],[k_matrix[1,2],0]]),np.array([k_matrix[2,0:2],[0,0]])]
                    k_reduced1 = k_matrix[0:2,0:2] - np.matmul(B,(1/k_matrix[2,2])*C) # matrix for before the interface
                    k_matrix1 = (k2*Ac)*np.array([[1/step2,-1/step2,0],[-1/step2,1/step2,0],[0,0,1/(3*step2)]])
                    k_matrix2 = (h*Pc)*np.array([[step2/3,step2/6,step2/12],[step2/6,step2/3,step2/12],[step2/12,step2/12,step2/30]])
                    k_matrix = k_matrix1 + k_matrix2
                    B,C = [np.array([[k_matrix[0,2],0],[k_matrix[1,2],0]]),np.array([k_matrix[2,0:2],[0,0]])]
                    k_reduced2 = k_matrix[0:2,0:2] - np.matmul(B,(1/k_matrix[2,2])*C) # matrix for after the interface
                    for j in range(1,i):
                        if j < SD[i-1][0][0]+1: # if j == index of discontinuity
                            K[j-1:j+1,j-1:j+1] += k_reduced1
                        else:
                            K[j-1:j+1,j-1:j+1] += k_reduced2
                    if BCs == (0,0): # case 1
                        K[0,0:2],K[i-1,i-2:i] = ((1,0),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    elif BCs == (1,0): # case 2
                        K[0,0:2],K[i-1,i-2:i] = ((k_reduced1[0,0],k_reduced1[0,1]),(0,1)) # instead of penalty method just defines the T0,TL as a function of one variable.
                    F[0],F[i-1] = [T0,TL]
                    B = np.linalg.solve(K,F)
                    case12_Tx_analytic, case12_Qdx_analytic = case12_analytic_setup(0,Disc[1],L,Disc_vals[0],a,k1,k2,Ac,Ac,T0,T_0,TL,Ta,BCs,Disc)
                    ind_of_interest,point_of_interest = [SD[i-1][0][0],SD[i-1][1][SD[i-1][0][0]]] # relatively in the middle # ind_of_interest,point_of_interest = [(i-1)//2,SD[i-1][1][(i-1)//2]] # relatively in the middle
                    TL_an, Qdx_an = [case12_Tx_analytic(point_of_interest),case12_Qdx_analytic(point_of_interest)]
                    T_err = float(log(abs((B[ind_of_interest] - TL_an)/TL_an),10))
                    for j in range(1,i):
                        step = SD[i-1][1][j]-SD[i-1][1][j-1]
                        if j < SD[i-1][0][0]+1:
                            Qdx4[j-1] = (-k1*Ac)*(((144*B[j])-(144*B[j-1])+(72*(step**2)*(Disc_vals[0]**2)*B[j])+(6*(step**4)*(Disc_vals[0]**4)*B[j]))/((144*step)+(24*(step**3)*(Disc_vals[0]**2))))
                        elif j > SD[i-1][0][0]:
                            Qdx4[j-1] = (-k2*Ac)*(((144*B[j])-(144*B[j-1])+(72*(step**2)*(a**2)*B[j])+(6*(step**4)*(a**4)*B[j]))/((144*step)+(24*(step**3)*(a**2))))
                    Qdx_err = float(log(abs((Qdx4[ind_of_interest-1] - Qdx_an)/Qdx_an),10))
                    VD[p][a][i-1] = {"xvals":SD[i-1][1],"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"HeatTran4":Qdx4,"T_error":T_err,"Qdx_err":Qdx_err,"Tx_eq":case12_Tx_analytic,"Qdx_eq":case12_Qdx_analytic}

# ######## Graphs Comparing FEM to Analytical
## Case 1 & 2 Temperature Plot
points = [("FDM",2),("FDM",4),("FEM",1),("FEM",2)]
alphas = [0.9,2.75,3.25,4.75]
sams = [8] # [4,8,16,32]
xvals_an = np.linspace(0,L,101)
if len(Disc) == 1:
    if BCs == (0,0): # case 1
        plt.title("Case 1 Temperature vs. Position")
    elif BCs == (1,0):
        plt.title("Case 2 Temperature vs. Position")
elif len(Disc) == 2:
    if BCs == (0,0): # case 1
        plt.title("Case 1 Temperature vs. Position, a_left = {}".format(Disc_vals[0]))
    elif BCs == (1,0):
        plt.title("Case 2 Temperature vs. Position, a_left = {}".format(Disc_vals[0]))
for p in points:
    for a in alphas:
        if p == ("FDM",2):
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"],VD[p][a][i]["Bvector"],label="FDM 2nd a,n={},{}".format(a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        elif p == ("FDM",4):
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"],VD[p][a][i]["Bvector"],label="FDM 4th a,n={},{}".format(a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        elif p[0] != "FDM":
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"],VD[p][a][i]["Bvector"],label="FEM p,a,n={},{},{}".format(p[1],a,i),color=(np.random.random(), np.random.random(), np.random.random()))
for a in alphas:
    plt.plot(xvals_an,[VD[p][a][i]["Tx_eq"](xval) for xval in xvals_an],label="analytical a={}".format(a),color="black")
plt.legend()
plt.xlabel("Position (cm)")
plt.ylabel("Temperature (C)")
plt.grid()
plt.show()
print()

# ######## Graphs Comparing FEM to Analytical
## Case 1 & 2 Rate of Heat Transfer Plot
points = [("FDM",2),("FDM",4),("FEM",1),("FEM",2)]
alphas = [0.9,2.75,3.25,4.75]
sams = [8] # [4,8,16,32]
xvals_an = np.linspace(0,L,101)
if len(Disc) == 1:
    if BCs == (0,0): # case 1
        plt.title("Case 1 Rate of Heat Transfer vs. Position")
    elif BCs == (1,0):
        plt.title("Case 2 Rate of Heat Transfer vs. Position")
elif len(Disc) == 2:
    if BCs == (0,0): # case 1
        plt.title("Case 1 Rate of Heat Transfer vs. Position, a_left = {}".format(Disc_vals[0]))
    elif BCs == (1,0):
        plt.title("Case 2 Rate of Heat Transfer vs. Position, a_left = {}".format(Disc_vals[0]))
for p in points:
    for a in alphas:
        if p == ("FDM",2):
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"][1:],VD[p][a][i]["HeatTran2"],label="Taylor2 FDM a,n={},{}".format(a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        if p == ("FDM",4):
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"][1:],VD[p][a][i]["HeatTran4"],label="Taylor4 FDM a,n={},{}".format(a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        elif p == ("FEM",1):
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"][1:],VD[p][a][i]["HeatTran2"],label="Taylor2 FEM p,a,n={},{},{}".format(p[1],a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        elif p == ("FEM",2):
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"][1:],VD[p][a][i]["HeatTran4"],label="Taylor4 FEM p,a,n={},{},{}".format(p[1],a,i),color=(np.random.random(), np.random.random(), np.random.random()))
for a in alphas:
    plt.plot(xvals_an,[VD[p][a][i]["Qdx_eq"](xval) for xval in xvals_an],label="analytical a={}".format(a),color="black")
plt.legend()
plt.xlabel("Position (cm)")
plt.ylabel("Rate of Heat Transfer (cal/s)")
plt.grid()
plt.show()
print()

######## Graphs Comparing FDM to Analytical Convergence
## Case 1 & 2 Temperature Convergence Plot
points = [("FDM",2),("FDM",4),("FEM",1),("FEM",2)]
alphas = [0.9,2.75,3.25,4.75]
sams = [4,8,16,32,64,128]
xvals_an = np.linspace(0,L,101)
if BCs == (0,0): # case 1
    plt.title("Case 1 Convergence of Temperature at T({}) for a={}".format(round(VD[p][a][sams[0]]["POI"],3),alphas))
elif BCs == (1,0):
    plt.title("Case 2 Convergence of Temperature at T({}) for a={}".format(round(VD[p][a][sams[0]]["POI"],3),alphas))
for p in points:
    for a in alphas:
        col1 = (np.random.random(), np.random.random(), np.random.random())
        T_err_vals = np.array([VD[p][a][sams[0]]["T_error"]])
        xvals_log = np.array([float(-log(L/i,10)) for i in sams])
        plt.plot(xvals_log[0],VD[p][a][sams[0]]["T_error"],"o",color=col1)
        for i,j in enumerate(sams[1:]):
            T_err = VD[p][a][j]["T_error"]
            T_err_vals = np.append(T_err_vals,T_err)
            plt.plot(xvals_log[i+1],T_err,"o",color=col1)
        Am1 = np.vstack([xvals_log, np.ones(len(xvals_log))]).T
        LS_coef1 = np.linalg.lstsq(Am1, T_err_vals,rcond=None)[0]
        xvals_LS,Nvals1, = [np.linspace(xvals_log[0],xvals_log[len(sams)-1],11),[]]
        for i in xvals_LS:
            Nvals1.append(sum([LS_coef1[k]*(i**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]))
        if p == ("FDM",2):
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="FDM 2nd a,B={},{}".format(a,round(abs(LS_coef1[0]),3)),color=col1)
        elif p == ("FDM",4):
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="FDM 4th a,B={},{}".format(a,round(abs(LS_coef1[0]),3)),color=col1)
        elif p[0] != "FDM":
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="FEM p,a={},{} ROC ={}".format(p[1],a,round(abs(LS_coef1[0]),3)),color=col1)
plt.legend()
plt.xlabel("-log10(dx)")
plt.ylabel("log10(e_abs)")
plt.grid()
plt.show()
print()

## Case 1 & 2 Rate of Heat Transfer Convergence Plot
points = [("FDM",2),("FDM",4),("FEM",1),("FEM",2)]
alphas = [0.9,2.75,3.25,4.75]
sams = [4,8,16,32,64,128]
xvals_an = np.linspace(0,L,101)
if BCs == (0,0): # case 1
    plt.title("Case 1 Convergence of Rate of Heat Transfer for a={}".format(alphas))
elif BCs == (1,0):
    plt.title("Case 2 Convergence of Rate of Heat Transfer for a={}".format(alphas))
for p in points:
    for a in alphas:
        col1 = (np.random.random(), np.random.random(), np.random.random())
        Qdx_err_vals = np.array([VD[p][a][sams[0]]["Qdx_err"]])
        xvals_log = np.array([float(-log(L/i,10)) for i in sams])
        plt.plot(xvals_log[0],VD[p][a][sams[0]]["Qdx_err"],"o",color=col1)
        for i,j in enumerate(sams[1:]):
            Qdx = VD[p][a][j]["Qdx_err"]
            Qdx_err_vals = np.append(Qdx_err_vals,Qdx)
            plt.plot(xvals_log[i+1],Qdx,"o",color=col1)
        Am1 = np.vstack([xvals_log, np.ones(len(xvals_log))]).T
        LS_coef1 = np.linalg.lstsq(Am1, Qdx_err_vals,rcond=None)[0]
        xvals_LS,Nvals1= [np.linspace(xvals_log[0],xvals_log[len(sams)-1],11),[]]
        for i in xvals_LS:
            Nvals1.append(sum([LS_coef1[k]*(i**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]))
        if p == ("FDM",2):
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="Taylor2 FDM a={} ROC={}".format(a,round(abs(LS_coef1[0]),3)),color=col1)
        elif p == ("FDM",4):
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="Taylor4 FDM a={} ROC={}".format(a,round(abs(LS_coef1[0]),3)),color=col1)
        elif p == ("FEM",1):
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="Taylor2 FEM p,a={},{} ROC ={}".format(p[1],a,round(abs(LS_coef1[0]),3)),color=col1)
        elif p == ("FEM",2):
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="Taylor4 FEM p,a={},{} ROC ={}".format(p[1],a,round(abs(LS_coef1[0]),3)),color=col1)
plt.legend()
plt.xlabel("-log10(dx)")
plt.ylabel("log10(e_abs)")
plt.grid()
plt.show()
print()