# Camden Hill
# 1D Thermal FEA
from sympy import log
import numpy as np
import matplotlib.pyplot as plt

def case12_analytic_setup(x0,xL,alpha1,u0,uL):
    matrix = np.array([[np.cos(alpha1*x0),np.sin(alpha1*x0)],[np.cos(alpha1*xL),np.sin(alpha1*xL)]])
    vector = np.array([u0,uL])
    c1,c2 = np.linalg.solve(matrix,vector)
    def case1_ux_analytic(x):
        return (c1*np.cos(alpha1*x)) + (c2*np.sin(alpha1*x))
    def case1_u_x_analytic(x):
        return (-c1*alpha1*np.sin(alpha1*x)) + (c2*alpha1*np.cos(alpha1*x))
    return case1_ux_analytic, case1_u_x_analytic

#### Problem Coefficients
h = 0.4             # Vibration Transfer Coefficient 
R = 0.1             # Cross section radius
Ac = np.pi * R**2   # Cross sectional area
Pc = 2 * np.pi * R      # Cross sectional perimeter
L = 1               # Bar length
As = 2*np.pi*R*L
# k = (h*2)/((a**2)*R)

# Case paramaters
u0 = 0
uL = 100

######## Finite Element Method
points = [("FDM",2),("FEM",1)] # [("FDM",2),("FEM",1)]
################ For Forced Frequency plots
# alphas = np.linspace((0.1)**0.5,(1000)**0.5,1000)
# sams = [4,8,16,32,64,128,256] # [4,8,16,32,64,128,256]
################ For Forced Frequency plots
alphas = [0.9,2.75,3.25,4.75] # [0.9,2.75,3.25,4.75] [0.29,0.50,1.35,2.75,4.75,9.15]
sams = [4,8,16,32,64,128] # [2,4,8] [4,8,16,32,64,128]
VD = {}
BCs = (0,0)

for p in points: # works only on sample size greater than 4
    VD[p] = {}
    for a in alphas:
        VD[p][a] = {}
        if p == ("FDM",2):
            for i in sams:
                step,k = [L/i,2-((a**2)*((L/i)**2))]
                i += 1
                K,F,udx = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                if BCs == (0,0): # case 1
                    K[0,0],K[i-1,i-1] = (1,1)
                # elif BCs == (1,0): # case 2
                #     K[0,0:2],K[i-1,i-1] = ((k/2,-1),1)
                for j in range(2,i):
                    K[j-1,j-2:j+1] = (-1,k,-1)
                F[0],F[i-1] = [u0,uL]
                B = np.linalg.solve(K,F)
                case1_ux_analytic, case1_u_x_analytic = case12_analytic_setup(0,L,a,u0,uL)
                ind_of_interest,point_of_interest = [(3*(i-1))//4,0.75*L] # at the 3/4ths point
                u_an, udx_an = [case1_ux_analytic(point_of_interest), case1_u_x_analytic(point_of_interest)]
                u_err = float(log(abs((B[ind_of_interest] - u_an)/u_an),10))
                ################ For Forced Frequency plots
                # ind_of_interest,point_of_interest = [-1,L] # The last index
                # u_an, udx_an = [case1_ux_analytic(point_of_interest), case1_u_x_analytic(point_of_interest)]
                # u_err = 1
                ################ For Forced Frequency plots
                for j in range(1,i):
                    udx[j-1] = ((B[j]-B[j-1])/step) - (0.5*step*(a**2)*B[j])
                udx_err = float(log(abs((udx[ind_of_interest-1] - udx_an)/udx_an),10))
                VD[p][a][i-1] = {"xvals":np.linspace(0,L,i),"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"u_dx":udx,"u_err":u_err,"udx_err":udx_err,"u_eq":case1_ux_analytic,"udx_eq":case1_u_x_analytic}
        elif p == ("FEM",1):
            for i in sams:
                step = L/i
                k0,k1 = [((2/step))-((a**2)*(2/3)*step),((-1/step))-((a**2)*(step/6))]
                i += 1
                K,F,udx = [np.zeros((i,i)),np.zeros((i)),np.zeros((i-1))]
                if BCs == (0,0): # case 1
                    K[0,0],K[i-1,i-1] = (1,1)
                for j in range(2,i):
                    K[j-1,j-2:j+1] = (k1,k0,k1)
                F[0],F[i-1] = [u0,uL]
                B = np.linalg.solve(K,F)
                case1_ux_analytic, case1_u_x_analytic = case12_analytic_setup(0,L,a,u0,uL)
                ind_of_interest,point_of_interest = [(3*(i-1))//4,0.75*L] # at the 3/4ths point
                u_an, udx_an = [case1_ux_analytic(point_of_interest), case1_u_x_analytic(point_of_interest)]
                u_err = float(log(abs((B[ind_of_interest] - u_an)/u_an),10))
                ################ For Forced Frequency plots
                # ind_of_interest,point_of_interest = [-1,L] # The last index
                # u_an, udx_an = [case1_ux_analytic(point_of_interest), case1_u_x_analytic(point_of_interest)]
                # u_err = 1
                ################ For Forced Frequency plots
                for j in range(1,i):
                    udx[j-1] = ((B[j]-B[j-1])/step) - (0.5*step*(a**2)*B[j])
                udx_err = float(log(abs((udx[ind_of_interest-1] - udx_an)/udx_an),10))
                VD[p][a][i-1] = {"xvals":np.linspace(0,L,i),"POI":point_of_interest,"Kmatrix":K,"Fvector":F,"Bvector":B,"u_dx":udx,"u_err":u_err,"udx_err":udx_err,"u_eq":case1_ux_analytic,"udx_eq":case1_u_x_analytic}


# ######## Graphs Comparing FEM to Analytical
## Case 1 & 2 Vibration Plot
points = [("FDM",2),("FEM",1)]
alphas = [0.9,2.75,3.25,4.75]
sams = [8] # [4,8,16,32]
xvals_an = np.linspace(0,L,101)
# if len(Disc) == 1:
if BCs == (0,0): # case 1
    plt.title("Vibration vs. Position")
# elif BCs == (1,0):
#     plt.title("Case 2 Vibration vs. Position")
# elif len(Disc) == 2:
#     if BCs == (0,0): # case 1
#         plt.title("Case 1 Vibration vs. Position, a_left = {}".format(Disc_vals[0]))
#     elif BCs == (1,0):
#         plt.title("Case 2 Vibration vs. Position, a_left = {}".format(Disc_vals[0]))
for p in points:
    for a in alphas:
        if p == ("FDM",2):
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"],VD[p][a][i]["Bvector"],label="FDM 2nd a,n={},{}".format(a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        # elif p == ("FDM",4):
        #     for i in sams:
        #         plt.plot(VD[p][a][i]["xvals"],VD[p][a][i]["Bvector"],label="FDM 4th a,n={},{}".format(a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        elif p[0] != "FDM":
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"],VD[p][a][i]["Bvector"],label="FEM p,a,n={},{},{}".format(p[1],a,i),color=(np.random.random(), np.random.random(), np.random.random()))
for a in alphas:
    plt.plot(xvals_an,[VD[p][a][i]["u_eq"](xval) for xval in xvals_an],label="analytical a={}".format(a),color="black")
plt.legend()
plt.xlabel("Position (cm)")
plt.ylabel("Vibration")
plt.grid()
plt.show()
print()

# ######## Graphs Comparing FEM to Analytical
## Case 1 & 2 Rate of Vibration Transfer Plot
points = [("FDM",2),("FEM",1)]
alphas = [0.9,2.75,3.25,4.75]
sams = [8] # [4,8,16,32]
xvals_an = np.linspace(0,L,101)
# if len(Disc) == 1:
if BCs == (0,0): # case 1
    plt.title("Rate of Vibration vs. Position")
# elif BCs == (1,0):
#     plt.title("Case 2 Rate of Vibration Transfer vs. Position")
# elif len(Disc) == 2:
#     if BCs == (0,0): # case 1
#         plt.title("Case 1 Rate of Vibration Transfer vs. Position, a_left = {}".format(Disc_vals[0]))
#     elif BCs == (1,0):
#         plt.title("Case 2 Rate of Vibration Transfer vs. Position, a_left = {}".format(Disc_vals[0]))
for p in points:
    for a in alphas:
        if p == ("FDM",2):
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"][1:],VD[p][a][i]["u_dx"],label="FDM a,n={},{}".format(a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        # if p == ("FDM",4):
        #     for i in sams:
        #         plt.plot(VD[p][a][i]["xvals"][1:],VD[p][a][i]["HeatTran4"],label="Taylor4 FDM a,n={},{}".format(a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        elif p == ("FEM",1):
            for i in sams:
                plt.plot(VD[p][a][i]["xvals"][1:],VD[p][a][i]["u_dx"],label="FEM p,a,n={},{},{}".format(p[1],a,i),color=(np.random.random(), np.random.random(), np.random.random()))
        # elif p == ("FEM",2):
        #     for i in sams:
        #         plt.plot(VD[p][a][i]["xvals"][1:],VD[p][a][i]["HeatTran4"],label="Taylor4 FEM p,a,n={},{},{}".format(p[1],a,i),color=(np.random.random(), np.random.random(), np.random.random()))
for a in alphas:
    plt.plot(xvals_an,[VD[p][a][i]["udx_eq"](xval) for xval in xvals_an],label="analytical a={}".format(a),color="black")
plt.legend()
plt.xlabel("Position (cm)")
plt.ylabel("Rate of Vibration")
plt.grid()
plt.show()
print()

######## Graphs Comparing FDM to Analytical Convergence
## Case 1 & 2 Vibration Convergence Plot
points = [("FDM",2),("FEM",1)]
alphas = [0.9,2.75,3.25,4.75]
sams = [4,8,16,32,64,128]
xvals_an = np.linspace(0,L,101)
if BCs == (0,0): # case 1
    plt.title("Convergence of Vibration at u({}) for a={}".format(round(1/(2*np.pi),3),alphas)) # 1/(2*np.pi) VD[p][a][sams[0]]["POI"]
# elif BCs == (1,0):
#     plt.title("Case 2 Convergence of Vibration at T({}) for a={}".format(round(VD[p][a][sams[0]]["POI"],3),alphas))
for p in points:
    for a in alphas:
        col1 = (np.random.random(), np.random.random(), np.random.random())
        T_err_vals = np.array([VD[p][a][sams[0]]["u_err"]])
        xvals_log = np.array([float(-log(L/i,10)) for i in sams])
        plt.plot(xvals_log[0],VD[p][a][sams[0]]["u_err"],"o",color=col1)
        for i,j in enumerate(sams[1:]):
            T_err = VD[p][a][j]["u_err"]
            T_err_vals = np.append(T_err_vals,T_err)
            plt.plot(xvals_log[i+1],T_err,"o",color=col1)
        Am1 = np.vstack([xvals_log, np.ones(len(xvals_log))]).T
        LS_coef1 = np.linalg.lstsq(Am1, T_err_vals,rcond=None)[0]
        xvals_LS,Nvals1, = [np.linspace(xvals_log[0],xvals_log[len(sams)-1],11),[]]
        for i in xvals_LS:
            Nvals1.append(sum([LS_coef1[k]*(i**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]))
        if p == ("FDM",2):
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="FDM 2nd a={} ROC={}".format(a,round(abs(LS_coef1[0]),3)),color=col1)
        # elif p == ("FDM",4):
        #     plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="FDM 4th a,ROC={},{}".format(a,round(abs(LS_coef1[0]),3)),color=col1)
        elif p[0] != "FDM":
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="FEM p,a={},{} ROC={}".format(p[1],a,round(abs(LS_coef1[0]),3)),color=col1)
plt.legend()
plt.xlabel("-log10(dx)")
plt.ylabel("log10(e_abs)")
plt.grid()
plt.show()
print()

## Case 1 & 2 Rate of Vibration Transfer Convergence Plot
points = [("FDM",2),("FEM",1)]
alphas = [0.9,2.75,3.25,4.75]
sams = [4,8,16,32,64,128]
xvals_an = np.linspace(0,L,101)
if BCs == (0,0): # case 1
    plt.title("Convergence of Rate of Vibration for a={}".format(alphas))
# elif BCs == (1,0):
#     plt.title("Case 2 Convergence of Rate of Vibration Transfer for a={}".format(alphas))
for p in points:
    for a in alphas:
        col1 = (np.random.random(), np.random.random(), np.random.random())
        udx_err_vals = np.array([VD[p][a][sams[0]]["udx_err"]])
        xvals_log = np.array([float(-log(L/i,10)) for i in sams])
        plt.plot(xvals_log[0],VD[p][a][sams[0]]["udx_err"],"o",color=col1)
        for i,j in enumerate(sams[1:]):
            Qdx = VD[p][a][j]["udx_err"]
            udx_err_vals = np.append(udx_err_vals,Qdx)
            plt.plot(xvals_log[i+1],Qdx,"o",color=col1)
        Am1 = np.vstack([xvals_log, np.ones(len(xvals_log))]).T
        LS_coef1 = np.linalg.lstsq(Am1, udx_err_vals,rcond=None)[0]
        xvals_LS,Nvals1= [np.linspace(xvals_log[0],xvals_log[len(sams)-1],11),[]]
        for i in xvals_LS:
            Nvals1.append(sum([LS_coef1[k]*(i**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]))
        if p == ("FDM",2):
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="FDM a={} ROC={}".format(a,round(abs(LS_coef1[0]),3)),color=col1)
        # elif p == ("FDM",4):
        #     plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="Taylor4 FDM a={} ROC={}".format(a,round(abs(LS_coef1[0]),3)),color=col1)
        elif p == ("FEM",1):
            plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="FEM p,a={},{} ROC={}".format(p[1],a,round(abs(LS_coef1[0]),3)),color=col1)
        # elif p == ("FEM",2):
        #     plt.plot(xvals_LS,Nvals1,linestyle="dashed",label="Taylor4 FEM p,a={},{} ROC={}".format(p[1],a,round(abs(LS_coef1[0]),3)),color=col1)
plt.legend()
plt.xlabel("-log10(dx)")
plt.ylabel("log10(e_abs)")
plt.grid()
plt.show()
print()

# ## Frequency Sweep Plot
# points = [("FEM",1)]
# alphas = np.linspace((0.1)**0.5,(1000)**0.5,1000)
# sams = [4,8,16,32,64,128,256]
# if BCs == (0,0): # case 1
#     plt.title("Frequency Sweep Plot (|u'(L)| vs a) for Sample Sizes = {}".format(sams))
# an_vals = []
# for a in alphas:
#     an_vals.append(abs(VD[("FEM",1)][a][sams[0]]["udx_eq"](VD[("FEM",1)][a][sams[0]]["POI"])))
# plt.plot(alphas,an_vals,label="Analytical",color="black")
# for p in points:
#     for i in sams:
#         FDM2_vals = []
#         for a in alphas:
#             FDM2_vals.append(abs(VD[p][a][i]["u_dx"][VD[p][a][i]["POI"]]))
#         if p == ("FDM",2):
#             plt.plot(alphas,FDM2_vals,label="FDM 2 n={}".format(i),color=(np.random.random(), np.random.random(), np.random.random()))
#         elif p == ("FEM",1):
#             plt.plot(alphas,FDM2_vals,label="FEM p,n={},{}".format(p[1],i),color=(np.random.random(), np.random.random(), np.random.random()))
# plt.legend()
# plt.xlabel("alphas")
# plt.ylabel("|u'(L)|")
# plt.ylim((0,50000))
# plt.grid()
# plt.show()
# print()