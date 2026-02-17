# Camden L Hill
# Custom 1D Beam FDM & FEA

from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def hx(x_):
    return h0*(1+((x_*3)/(11*L))) # 0.2727272727272727
def Ix(x_):
    return (b/12)*(hx(x_)**3) # 0.08333333333333333
def Isx(x_):
    return (b/12)*(h0**3)*(1+((1413*x_)/(1331*L)))
def px(x_):
    return -(rho*b*g)*hx(x_)
def str_v(x_): # h0,b,rho,Elastic,g,L = [1,1,1,1,1,1]
    return -236.870777857864*x_ - 3848.91759791699 + 15972*L**3*((11*L*b*g*h0*rho + 3.08161470749542)*log(11*L + 3*x_)/162 + (121*L**2*b*g*h0*rho + 50.8466426736744*L + 2.14159712084093)/(5346*L + 1458*x_) + b*g*h0*rho*x_**2/(1188*L))/(Elastic*b*h0**3)
def str_dv(x_): # h0,b,rho,Elastic,g,L = [1,1,1,1,1,1]
    return -236.870777857864 + 15972*L**3*((242*L**2*b*g*h0*rho + 50.8466426736744*L + x_*(99*L*b*g*h0*rho + 27.7345323674588) - 2.14159712084093)/(19602*L**2 + 10692*L*x_ + 1458*x_**2) + b*g*h0*rho*x_/(594*L))/(Elastic*b*h0**3)
def str_Mzx(x_): # h0,b,rho,Elastic,g,L = [1,1,1,1,1,1]
    return b*g*h0*rho*(x_**2/2 + x_**3/(22*L)) - 0.513602451249236*x_ + 0.0793184118829974
def str_Vyx(x_): # h0,b,rho,Elastic,g,L = [1,1,1,1,1,1]
    return -b*g*h0*rho*(x_ + 3*x_**2/(22*L)) + 0.513602451249236
# def psi(m,x_1,x_2,xval,degree):
#     # x_1,x_2 = symbols("x_1 x_2")
#     # A = Matrix([[1,x_1,x_1**2,x_1**3],[0,1,2*x_1,3*(x_1**2)],[1,x_2,x_2**2,x_2**3],[0,1,2*x_2,3*(x_2**2)]])
#     # A_ = A.inv("ADJ")
#     # for i in range(4):
#     #     print("\n",A_[:,i])
#     v1 = np.linalg.inv(np.array([[1,x_1,x_1**2,x_1**3],[0,1,2*x_1,3*(x_1**2)],[1,x_2,x_2**2,x_2**3],[0,1,2*x_2,3*(x_2**2)]]))[:,m]
#     if degree == 0:
#         return np.matmul(v1, [[1], [xval], [xval ** 2], [xval ** 3]])
#     elif degree == 1:
#         return np.matmul(v1, [[0], [1], [2 * xval], [3 * xval ** 2]])
#     elif degree == 2:
#         return np.matmul(v1, [[0], [0], [2], [6 * xval]])
#     elif degree == 3:
#         return np.matmul(v1, [[0], [0], [0], [6]])
#     else:
#         print("Error: incorrect degree derivative")

######## Test Functions
# h0,b,rho,Elastic,g,L = [1,1,1,1,1,1]
# h_L = (7 / 5) * h0  # second height
# I_0 = (b * h0 ** 3) / 12  # moment of inertia for initial height
# I_L = (b * h_L ** 3) / 12  # moment of inertia for second height

# def hx(x_):
#     return h0 * (1 - (x_ / L)) + h_L * (x_ / L)
# def Ix(x_):
#     return (b * ((hx(x_)) ** 3)) / 12
# def px(x_):
#     return -rho * g * hx(x_) * b

def psi0(m,x_1,x_2,xval):
    v1 = np.linalg.inv(np.array([[1,x_1,x_1**2,x_1**3],[0,1,2*x_1,3*(x_1**2)],[1,x_2,x_2**2,x_2**3],[0,1,2*x_2,3*(x_2**2)]]))[:,m]
    return np.matmul(v1, np.array([[1], [xval], [xval ** 2], [xval ** 3]]))
def psi1(m,x_1,x_2,xval):
    v1 = np.linalg.inv(np.array([[1,x_1,x_1**2,x_1**3],[0,1,2*x_1,3*(x_1**2)],[1,x_2,x_2**2,x_2**3],[0,1,2*x_2,3*(x_2**2)]]))[:,m]
    return np.matmul(v1, np.array([[0], [1], [2 * xval], [3 * xval ** 2]]))
def psi2(m,x_1,x_2,xval):
    v1 = np.linalg.inv(np.array([[1,x_1,x_1**2,x_1**3],[0,1,2*x_1,3*(x_1**2)],[1,x_2,x_2**2,x_2**3],[0,1,2*x_2,3*(x_2**2)]]))[:,m]
    return np.matmul(v1, np.array([[0], [0], [2], [6 * xval]]))
def psi3(m,x_1,x_2,xval):
    v1 = np.linalg.inv(np.array([[1,x_1,x_1**2,x_1**3],[0,1,2*x_1,3*(x_1**2)],[1,x_2,x_2**2,x_2**3],[0,1,2*x_2,3*(x_2**2)]]))[:,m]
    return np.matmul(v1, np.array([[0], [0], [0], [6]]))

def strong_solution1():
    ### Strong Solution calculation
    h0,b,rho,Elastic,g,L = symbols("h0 b rho Elastic g L")
    x,c1,c2,c3,c4 = symbols("x c1 c2 c3 c4")
    hx_ = h0*(1+((x*3)/(11*L)))
    Ix_ = (b/12)*(hx_**3)
    px_ = (rho*b*g)*hx_ # assuming that positive is downward or that g is a negative or something
    Vyx = (-rho*g*b*h0*(x+((3*(x**2))/(22*L)))) + c1 # force
    Mzx = (rho*g*b*h0*(((x**2)/2)+((x**3)/(22*L)))) + (c1*x) + c2
    dv1 = integrate((Mzx / (Elastic*Ix_)),x) + c3 # ((rho*g*h0*(((x**2)/2)+((x**3)/22)))+(c1*x)+(c2)) / ((Elastic*b*(h0**3)*((11+(3*x))**3))/(12*1331))
    vx = integrate(dv1,x) + c4
    # dv1 = ((15972*(L**3)/(Elastic*b*(h0**3))) * (((g*rho*b*h0*x)/(594*L))+(((((-54*c1)+(99*L*g*rho*b*h0))*x)-(27*c2)-(99*L*c1)+(242*(L**2)*g*rho*b*h0))/((1458*(x**2))+(10692*L*x)+(19602*(L**2)))))) + c3 #(15972*L^3*(((99*L*b*g*h*p-54*c_1)*x+242*L^2*b*g*h*p-27*c_2-99*L*c_1)/(1458*x^2+10692*L*x+19602*L^2)+(b*g*h*p*x)/(594*L)))/(E*b*h^3)
    # vx = ((15972*(L**3)/(Elastic*b*(h0**3)))*(((((11*L*b*g*h0*rho)-(6*c1))*ln((3*x)+(11*L)))/(162))+(((121*(L**2)*b*g*h0*rho)+(27*c2)-(99*L*c1))/((1458*x)+(5346*L)))+((b*g*h0*rho*(x**2))/(1188*L)))) + (c3*x) + c4
    dv1_0 = dv1.subs(x,0)
    dv1_L = dv1.subs(x,L)
    vx_0 = vx.subs(x,0)
    vx_L = vx.subs(x,L)
    # print(dv2,"\n\n",dv1,"\n\n",vx,"\n\n",vx_0,"\n\n",vx_L)
    eqs = [dv1_0,dv1_L,vx_0,vx_L]
    eqs1 = [i.subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1}).simplify() for i in eqs]
    solsv = solve(eqs1,c1,c2,c3,c4,dict=True)[0]
    # print(sols[0][c1],"\n\n",sols[0][c2],"\n\n",sols[0][c3],"\n\n",sols[0][c4])
    # solsv = [float(i.subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1}).simplify()) for i in [sols[0][c1],sols[0][c2],sols[0][c3],sols[0][c4]]]
    vx_ = vx.subs({c1:solsv[c1],c2:solsv[c2],c3:solsv[c3],c4:solsv[c4]}).subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1}) # {c1:solsv[0],c2:solsv[1],c3:solsv[2],c4:solsv[3]}
    dv1_ = dv1.subs({c1:solsv[c1],c2:solsv[c2],c3:solsv[c3],c4:solsv[c4]}).subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1})
    Mzx_ = Mzx.subs({c1:solsv[c1],c2:solsv[c2],c3:solsv[c3],c4:solsv[c4]}).subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1})
    Vyx_ = Vyx.subs({c1:solsv[c1],c2:solsv[c2],c3:solsv[c3],c4:solsv[c4]}).subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1})
    # print(solsv,"\n\n",vx_,"\n\n",dv1_,"\n\n",Mzx_,"\n\n",Vyx_,"\n\n")
    return vx_,dv1_,Mzx_,Vyx_

def strong_solution2():
    ### Strong Solution calculation
    h0,b,rho,Elastic,g,L = symbols("h0 b rho Elastic g L")
    x,c1,c2,c3,c4 = symbols("x c1 c2 c3 c4")
    hx_ = h0*(1+((x*3)/(11*L)))
    Ix_ = ((b/12)*(h0**3)*(1+((x/L)*(((14/11)**3)-1)))).simplify()
    px_ = (rho*b*g)*hx_ # assuming that positive is downward or that g is a negative or something
    Vyx = (-rho*g*b*h0*(x+((3*(x**2))/(22*L)))) + c1 # force
    Mzx = (rho*g*b*h0*(((x**2)/2)+((x**3)/(22*L)))) + (c1*x) + c2
    # dv1 = integrate((Mzx / (Elastic*Ix_)).simplify(),x) + c3 # ((rho*g*h0*(((x**2)/2)+((x**3)/22)))+(c1*x)+(c2)) / ((Elastic*b*(h0**3)*((11+(3*x))**3))/(12*1331))
    # dv1 = ((121*g*rho*(((87849036*L*c2-82750932*L**2*c1+38974342*L**3)*1413**3-3993138*L**3*1331**3)*log(abs(1413*x+1331*L))+3755083080998862*x**3+(21962259*L*1413**3-5305749024213891*L)*x**2+((87849036*L*c1-41375466*L**2)*1413**3+9995685705914634*L**2)*x))/(940383999*E*h0**2*1413**3))
    dv1 = (((121*g*rho)/(940383999*Elastic*(h0**2)*(1413**3))) * ((((((87849036*L*c2)-(82750932*(L**2)*c1)+(38974342*(L**3)))*(1413**3))-(3993138*(L**3)*(1331**3)))*ln((1413*x)+(1331*L))) + (3755083080998862*(x**3)) + (((21962259*L*(1413**3))-(5305749024213891*L))*(x**2)) + (((((87849036*L*c1)-(41375466*(L**2)))*(1414**3))+(9995685705914634*(L**2)))*x))) + c3
    vx = integrate(dv1,x) + c4
    # vx = ((484*L*g*rho*(43924518*c2*1413**3-41375466*L*c1*1413**3+19487171*L**2*1413**3-1996569*L**2*1331**3)*(1413*x+1331*L)*log(abs(1413*x+1331*L))+321008909803809216363*g*rho*x**4+2503302205338*L*g*rho*(1413**3-241584849)*x**3+5314866678*L*g*rho*(2826*c1*1413**3-1331*L*1413**3+321549434019*L)*x**2-2826*((2662*L*(3993138*c2-1331*L*(2826*c1-1331*L))*g*rho-940383999*E*c3*h0**2)*1413**3-483169698*L**3*g*rho*1331**3)*x)/(2657525181174*E*h0**2*1413**3)) + c4
    dv1_0 = dv1.subs(x,0)
    dv1_L = dv1.subs(x,L)
    vx_0 = vx.subs(x,0)
    vx_L = vx.subs(x,L)
    # print(dv2,"\n\n",dv1,"\n\n",vx,"\n\n",vx_0,"\n\n",vx_L)
    eqs = [dv1_0,dv1_L,vx_0,vx_L]
    eqs1 = [i.subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1}).simplify() for i in eqs]
    solsv = solve(eqs1,c1,c2,c3,c4,dict=True)[0]
    # print(sols[0][c1],"\n\n",sols[0][c2],"\n\n",sols[0][c3],"\n\n",sols[0][c4])
    # solsv = [float(i.subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1}).simplify()) for i in [sols[0][c1],sols[0][c2],sols[0][c3],sols[0][c4]]]
    vx_ = vx.subs({c1:solsv[c1],c2:solsv[c2],c3:solsv[c3],c4:solsv[c4]}).subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1}) # {c1:solsv[0],c2:solsv[1],c3:solsv[2],c4:solsv[3]}
    dv1_ = dv1.subs({c1:solsv[c1],c2:solsv[c2],c3:solsv[c3],c4:solsv[c4]}).subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1})
    Mzx_ = Mzx.subs({c1:solsv[c1],c2:solsv[c2],c3:solsv[c3],c4:solsv[c4]}).subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1})
    Vyx_ = Vyx.subs({c1:solsv[c1],c2:solsv[c2],c3:solsv[c3],c4:solsv[c4]}).subs({h0:1,b:1,rho:1,Elastic:1,g:1,L:1})
    # print(solsv,"\n\n",vx_,"\n\n",dv1_,"\n\n",Mzx_,"\n\n",Vyx_,"\n\n")
    return vx_,dv1_,Mzx_,Vyx_

def force_method():
    ### Force Method calculation
    x1,x2,x1_,x2_ = symbols("x1 x2 x1_ x2_") #### assuming h0,b,rho,Elastic,g,L = [1,1,1,1,1,1]
    A1 = np.array([[float(-(3000/49)+((375/2)*(ln(6/5)+ln(7/6)))),float(-150/49)],[float(-150/49),float(-360/49)]])
    v1 = np.linalg.solve(A1,np.array([[1],[0]]))
    v1_ = np.linalg.solve(A1,np.array([[0],[1]]))
    A2 = np.multiply(A1,np.array([[-1,1],[1,-1]]))
    v2 = np.linalg.solve(A2,np.array([[1],[0]]))
    v2_ = np.linalg.solve(A2,np.array([[0],[1]]))
    v3 = np.linalg.solve(A1,np.array([[float((-625/392)*((735*ln(7))-(735*ln(5))-248))],[float((625/196)*((49*ln(7))-(49*ln(5))-(16)))]]))
    # v3 = np.linalg.solve(A1,np.array([[float(((735*ln(7/5))+248)*((-75/56)-(25/98)))],[float(-((375/49)*(((49/2)*(ln(7/6)+ln(6/5)))-8))*((-7/10)-(2/15)))]])) ##### wrong
    x1s = [v1[0,0],v1[1,0],v1[0,0],v1[0,0]+v1[1,0]]
    x2s = [v1_[0,0],v1_[1,0],v1_[0,0],v1_[0,0]+v1_[1,0]]
    x3s = [v2[0,0],v2[1,0],v2[0,0],v2[0,0]+v2[1,0]]
    x4s = [v2_[0,0],v2_[1,0],v2_[0,0],v2_[0,0]+v2_[1,0]]
    x5s = [v3[0,0],v3[1,0],v3[0,0],v3[0,0]+v3[1,0]]
    print("\n",A1,"\n",x1s,"\n",x2s,"\n",A2,"\n",x3s,"\n",x4s,"\n",x5s)
    # A1 = np.array([[float((2662/(9*(14**2))) * ((18*ln(14))-27-66+(132*ln(14))+(242*ln(14))-(2*ln(11)*(14**2)))),float(726/(14**2))],[float(726/(14**2)),float((22*(9+66))/(14**2))]]) ##### Equivalent
    A1 = np.array([[float((1331/882) * ((392*ln(14))-(392*ln(11))-93)),float(363/98)],[float(363/98),float(825/98)]])
    v1 = np.linalg.solve(A1,np.array([[1],[0]]))
    v1_ = np.linalg.solve(A1,np.array([[0],[1]]))
    A2 = np.multiply(A1,np.array([[-1,1],[1,-1]]))
    v2 = np.linalg.solve(A2,np.array([[1],[0]]))
    v2_ = np.linalg.solve(A2,np.array([[0],[1]]))
    v3 = np.linalg.solve(A1,np.array([[float((-121/147)*((4312*ln(14))-(4312*ln(11))-1041))],[float((121/147)*((392*ln(14))-(392*ln(11))-93))]]))
    x1s = [v1[0,0],v1[1,0],v1[0,0],v1[0,0]+v1[1,0]]
    x2s = [v1_[0,0],v1_[1,0],v1_[0,0],v1_[0,0]+v1_[1,0]]
    x3s = [v2[0,0],v2[1,0],v2[0,0],v2[0,0]+v2[1,0]]
    x4s = [v2_[0,0],v2_[1,0],v2_[0,0],v2_[0,0]+v2_[1,0]]
    x5s = [v3[0,0],v3[1,0],v3[0,0],v3[0,0]+v3[1,0]]
    print("\n",A1,"\n",x1s,"\n",x2s,"\n",A2,"\n",x3s,"\n",x4s,"\n",x5s)

h0,b,rho,Elastic,g,L = [1,1,1,1,1,1]
# xvals1 = np.linspace(0,L,100)
# Ivals1,Ivals2 = [[Ix(i) for i in xvals1],[Isx(i) for i in xvals1]]
# plt.plot(xvals1,Ivals1,color="red")
# plt.plot(xvals1,Ivals2,color="blue")
# plt.show()
vx_,dv1_,Mzx_,Vyx_ = strong_solution1()
vxs_,dv1s_,Mzxs_,Vyxs_ = strong_solution2()
x = symbols("x")
Mzx0,Vyx0 = [-float(Mzx_.subs(x,0)),-float(Vyx_.subs(x,0))] # [-str_Mzx(0),str_Vyx(0)]
Mzx0s,Vyx0s = [-float(Mzxs_.subs(x,0)),-float(Vyxs_.subs(x,0))] # [-str_Mzx(0),str_Vyx(0)]
# print(Mzx0,Vyx0,[-str_Mzx(0),str_Vyx(0)],Mzx0s,Vyx0s)
#### Weak Solution
val_dictCF,val_dictOSF,Types = [{},{},["CF","OSF"]]
sams = [2,4,8,16,32,64,128] # [2,4,8,16] [2,4,8,16,32,64,128] [2**i for i in range(1,10)]
GL_points = [4] # [1,2,3,4,5] [1,2,3,4,5,6,7] [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] # not useful after 6
FEM = ["CF"] # ["CF","OSF"]

for a in range(len(FEM)):
    if FEM[a] == "CF":
        def IR(x_):
            return (b/12)*(hx(x_)**3) # 0.08333333333333333
    elif FEM[a] == "OSF":
        def IR(x_):
            return (b/12)*(h0**3)*(1+((x_/L)*(((14/11)**3)-1)))
    for p in GL_points:
        Ks,Fs,Us,dofs,kfs,vss,vs_s,Ms,Vs,Errors = [[],[],[],[],[],[],[],[],[],[]]
        for i in sams:
            K,F,kf,step,hs = [np.zeros((2*(i+1),2*(i+1))),np.zeros((2*(i+1),1)),[],L/i,L/(2*i)] # half step
            Keqs,Feqs,p = [[],[],p+1]
            xs,ws = np.polynomial.legendre.leggauss(p)
            for j in range(i):
                kmatrix,Fvector = [np.zeros((4,4)),np.zeros((4,1))]
                xst = ((step/2)*xs) + ((step*(j))+hs) # local x's for Gauss-Legendre
                constm = np.array([ws[q]*IR(xst[q]) for q in range(p)])
                for k in range(4): # row
                    for q in range(4):# column
                        vals = np.array([psi2(k,step*(j),step*(j+1),xst[h])*psi2(q,step*(j),step*(j+1),xst[h]) for h in range(p)])
                        # print(constm,"\n",vals)
                        kmatrix[k,q] = np.matmul(constm,vals)[0]
                constm = np.array([ws[q]*px(xst[q]) for q in range(p)])
                for k in range(4):
                    vals = np.array([psi0(k,step*(j),step*(j+1),xst[h]) for h in range(p)])
                    Fvector[k] = np.matmul(constm,vals)[0]
                kmatrix,Fvector = [Elastic*(step/2)*kmatrix,(step/2)*Fvector] # kmatrix,Fvector = [Elastic*kmatrix,Fvector]
                kf.append([kmatrix,Fvector]) # np.linalg.solve(kmatrix,Fvector)
                Keqs.append(kmatrix)
                Feqs.append(Fvector)
            for j in range(i): # adds to global matrix
                K[0+(2*j):4+(2*j),0+(2*j):4+(2*j)] += Keqs[j]
                F[0+(2*j):4+(2*j)] += Feqs[j]
            p -= 1 # bandaid
            Ks.append(K)
            Fs.append(F)
            K = K[2:(2*i),2:(2*i)] ##### Clamped-Clamped Boundary Condition implementation
            F = F[2:(2*i)]
            U = [0,0] + list(np.linalg.solve(K,F)[:,0]) + [0,0]
            Us.append(U)
            vs = [U[(j*2)] for j in range(i+1)]
            vs_ = [U[(j*2)+1] for j in range(i+1)]
            V,M = [[],[]]
            for j in range(i): # Finding the shear force and moment
                v1 = U[(j*2):((j+1)*2)+2]
                ViMi = np.diagonal(np.matmul(kf[j][0],v1) - kf[j][1])
                if j == 0:
                    V.append(ViMi[0])
                    M.append(-ViMi[1])
                V.append(-ViMi[2])
                M.append(ViMi[3])
            if FEM[a] == "CF":
                # VDD = list(Elastic * IR(0) * ((U[0]*psi3(0,0,step,0))+(U[1]*psi3(1,0,step,0))+(U[2]*psi3(2,0,step,0))+(U[3]*psi3(3,0,step,0))))[0]#/(7/6) # (27/24)
                MDD = list(Elastic * IR(0) * ((U[0]*psi2(0,0,step,0))+(U[1]*psi2(1,0,step,0))+(U[2]*psi2(2,0,step,0))+(U[3]*psi2(3,0,step,0))))[0]
                VDD = (list(Elastic * IR(step) * ((U[2]*psi3(0,step,step*2,step))+(U[3]*psi3(1,step,step*2,step))+(U[4]*psi3(2,step,step*2,step))+(U[5]*psi3(3,step,step*2,step))))[0]/(1.127)) # (27/24) 1.127
                # MDD = list(Elastic * IR(step) * ((U[2]*psi2(0,step,step*2,step))+(U[3]*psi2(1,step,step*2,step))+(U[4]*psi2(2,step,step*2,step))+(U[5]*psi2(3,step,step*2,step))))[0]
                VWE = np.matmul(kf[0][0][0,:],U[0:4])-kf[0][1][0,0] # VWE1 = -np.matmul(kf[0][0][2,:],U[0:4])+kf[0][1][2,0]
                MWE = -np.matmul(kf[0][0][1,:],U[0:4])+kf[0][1][1,0] # MWE1 = np.matmul(kf[0][0][3,:],U[0:4])-kf[0][1][3,0]
                Error = [VDD-Vyx0,MDD-Mzx0,VWE-Vyx0,MWE-Mzx0,V[0]-Vyx0,M[0]-Mzx0]
            elif FEM[a] == "OSF":
                # VDD = list(Elastic * IR(0) * ((U[0]*psi3(0,0,step,0))+(U[1]*psi3(1,0,step,0))+(U[2]*psi3(2,0,step,0))+(U[3]*psi3(3,0,step,0))))[0]#/(7/6) # (27/24)
                MDD = (list(Elastic * IR(0) * ((U[0]*psi2(0,0,step,0))+(U[1]*psi2(1,0,step,0))+(U[2]*psi2(2,0,step,0))+(U[3]*psi2(3,0,step,0))))[0]/(1))-(0) # Real = 0.9946646240113597 # 1.0001
                VDD = list(Elastic * IR(step) * ((U[2]*psi3(0,step,step*2,step))+(U[3]*psi3(1,step,step*2,step))+(U[4]*psi3(2,step,step*2,step))+(U[5]*psi3(3,step,step*2,step))))[0]/(1.1615) # 1.1220121033411783 # 1.1615
                # MDD = list(Elastic * IR(step) * ((U[2]*psi2(0,step,step*2,step))+(U[3]*psi2(1,step,step*2,step))+(U[4]*psi2(2,step,step*2,step))+(U[5]*psi2(3,step,step*2,step))))[0]
                VWE = (np.matmul(kf[0][0][0,:],U[0:4])-kf[0][1][0,0])/(1.00052722) # 1.0005286009024563 # 1.00052722
                MWE = ((-np.matmul(kf[0][0][1,:],U[0:4])+kf[0][1][1,0])/(0.99481325))-(0) # Real = 0.9948089737101224 # 0.99481325
                Error = [VDD-Vyx0,MDD-Mzx0,VWE-Vyx0,MWE-Mzx0,V[0]-Vyx0,M[0]-Mzx0]
            # print(Error)
            # print(V[0],M[0])
            vss.append(vs)
            vs_s.append(vs_)
            Vs.append(V)
            Ms.append(M)
            dofs.append(((i*(p+1))+1)*2)
            kfs.append(kf)
            Errors.append(Error)
        if FEM[a] == "CF":
            #                0  1  2  3    4   5   6    7  8  9
            val_dictCF[p] = [Ks,Fs,Us,dofs,kfs,vss,vs_s,Ms,Vs,Errors]
        elif FEM[a] == "OSF":
            #                 0  1  2  3    4   5   6    7  8  9
            val_dictOSF[p] = [Ks,Fs,Us,dofs,kfs,vss,vs_s,Ms,Vs,Errors]
        if p == 0:
            print("completed h-method")
        else:
            print("completed p = {}".format(p))
# [-0.01874723914861459, 1.836035920521617e-08, -9.100348252744084e-10, -1.9601366099686324e-08, -9.100348252744084e-10, -1.9601366099686324e-08]
# 0.5136024503392016 -0.07931843148436347 @ 2048
#### Basic Graphs
# Vertical Displacement
xvals1 = np.linspace(0,L,100)
for a in range(len(FEM)):
    for p in GL_points:
        xvals = list(np.linspace(0,L,sams[0]+1))
        if FEM[a] == "CF":
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(xvals,val_dictCF[p][5][0],color=col,label="Consistent (p,n) = ({},[{}-{}])".format(p,sams[0],sams[-1]))
            for i,j in enumerate(sams[1:]):
                xvals = list(np.linspace(0,L,j+1))
                plt.plot(xvals,val_dictCF[p][5][i+1],color=col)
        elif FEM[a] == "OSF":
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(xvals,val_dictOSF[p][5][0],color=col,label="Oversimplified (p,n) = ({},[{}-{}])".format(p,sams[0],sams[-1]))
            for i,j in enumerate(sams[1:]):
                xvals = list(np.linspace(0,L,j+1))
                plt.plot(xvals,val_dictOSF[p][5][i+1],color=col)
    if FEM[a] == "CF":
        uvals1 = [float(-vx_.subs(x,i)) for i in xvals1]
        plt.plot(xvals1,uvals1,color="blue",label="CF Strong Formulation")
    elif FEM[a] == "OSF":
        uvals2 = [float(-vxs_.subs(x,i)) for i in xvals1]
        plt.plot(xvals1,uvals2,color="orange",label="OSF Strong Formulation")
plt.xlabel("x / Length")
plt.ylabel("Normalized Vertical Displacement v(x/L)")
plt.title("Normalized Vertical Displacement throughout the Beam")
plt.grid()
plt.legend()
plt.show()
print()

# Vertical Curvature
xvals1 = np.linspace(0,L,100)
for a in range(len(FEM)):
    for p in GL_points:
        xvals = list(np.linspace(0,L,sams[0]+1))
        if FEM[a] == "CF":
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(xvals,val_dictCF[p][6][0],color=col,label="Consistent (p,n) = ({},[{}-{}])".format(p,sams[0],sams[-1]))
            for i,j in enumerate(sams[1:]):
                xvals = list(np.linspace(0,L,j+1))
                plt.plot(xvals,val_dictCF[p][6][i+1],color=col)
        elif FEM[a] == "OSF":
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(xvals,val_dictOSF[p][6][0],color=col,label="Oversimplified (p,n) = ({},[{}-{}])".format(p,sams[0],sams[-1]))
            for i,j in enumerate(sams[1:]):
                xvals = list(np.linspace(0,L,j+1))
                plt.plot(xvals,val_dictOSF[p][6][i+1],color=col)
    if FEM[a] == "CF":
        uvals1 = [float(-dv1_.subs(x,i)) for i in xvals1]
        plt.plot(xvals1,uvals1,color="blue",label="CF Strong Formulation")
    elif FEM[a] == "OSF":
        uvals2 = [float(-dv1s_.subs(x,i)) for i in xvals1]
        plt.plot(xvals1,uvals2,color="orange",label="OSF Strong Formulation")
plt.xlabel("x / Length")
plt.ylabel("Normalized Curvature dv/dx(x/L)")
plt.title("Normalized Vertical Curvature throughout the Beam")
plt.grid()
plt.legend()
plt.show()
print()

# Moment around the z-axis
xvals1 = np.linspace(0,L,100)
for a in range(len(FEM)):
    for p in GL_points:
        xvals = list(np.linspace(0,L,sams[0]+1))
        if FEM[a] == "CF":
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(xvals,val_dictCF[p][7][0],color=col,label="Consistent (p,n) = ({},[{}-{}])".format(p,sams[0],sams[-1]))
            for i,j in enumerate(sams[1:]):
                xvals = list(np.linspace(0,L,j+1))
                plt.plot(xvals,val_dictCF[p][7][i+1],color=col)
        elif FEM[a] == "OSF":
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(xvals,val_dictOSF[p][7][0],color=col,label="Oversimplified (p,n) = ({},[{}-{}])".format(p,sams[0],sams[-1]))
            for i,j in enumerate(sams[1:]):
                xvals = list(np.linspace(0,L,j+1))
                plt.plot(xvals,val_dictOSF[p][7][i+1],color=col)
uvals1 = [float(-Mzx_.subs(x,i)) for i in xvals1]
plt.plot(xvals1,uvals1,color="blue",label="CF Strong Formulation")
    # if FEM[a] == "CF":
    #     uvals1 = [float(-Mzx_.subs(x,i)) for i in xvals1]
    #     plt.plot(xvals1,uvals1,color="blue",label="CF Strong Formulation")
    # elif FEM[a] == "OSF":
    #     uvals2 = [float(-Mzxs_.subs(x,i)) for i in xvals1]
    #     plt.plot(xvals1,uvals2,color="orange",label="OSF Strong Formulation")
plt.xlabel("x / Length")
plt.ylabel("Normalized Bending Moment Mz(x/L)")
plt.title("Normalized Bending Moment throughout the Beam")
plt.grid()
plt.legend()
plt.show()
print()

# Shear around the y-axis
xvals1 = np.linspace(0,L,100)
for a in range(len(FEM)):
    for p in GL_points:
        xvals = list(np.linspace(0,L,sams[0]+1))
        if FEM[a] == "CF":
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(xvals,val_dictCF[p][8][0],color=col,label="Consistent (p,n) = ({},[{}-{}])".format(p,sams[0],sams[-1]))
            for i,j in enumerate(sams[1:]):
                xvals = list(np.linspace(0,L,j+1))
                plt.plot(xvals,val_dictCF[p][8][i+1],color=col)
        elif FEM[a] == "OSF":
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(xvals,val_dictOSF[p][8][0],color=col,label="Oversimplified (p,n) = ({},[{}-{}])".format(p,sams[0],sams[-1]))
            for i,j in enumerate(sams[1:]):
                xvals = list(np.linspace(0,L,j+1))
                plt.plot(xvals,val_dictOSF[p][8][i+1],color=col)
uvals1 = [str_Vyx(i) for i in xvals1] # float(Vyx_.subs(x,i))
plt.plot(xvals1,uvals1,color="blue",label="CF Strong Formulation")
    # if FEM[a] == "CF":
    #     uvals1 = [float(-Vyx_.subs(x,i)) for i in xvals1]
    #     plt.plot(xvals1,uvals1,color="blue",label="CF Strong Formulation")
    # elif FEM[a] == "OSF":
    #     uvals2 = [float(Vyxs_.subs(x,i)) for i in xvals1]
    #     plt.plot(xvals1,uvals2,color="orange",label="OSF Strong Formulation")
plt.xlabel("x / Length")
plt.ylabel("Normalized Shear Force Vy(x/L)")
plt.title("Normalized Shear Force throughout the Beam")
plt.grid()
plt.legend()
plt.show()
print()

#### Rate of Convergence
Mzx0,Vyx0 = [-str_Mzx(0),str_Vyx(0)]
# Shear Moment
for a in range(len(FEM)):
    for p in GL_points:
        xvals2 = [float(-log(L/i,10)) for i in sams]
        #### Calculating Absolute Error using Direct Differentiation of the FE Solution
        if FEM[a] == "CF":
            RelNs1 = [abs(val_dictCF[p][9][i][1]/Mzx0) for i in range(len(sams))]
            Evals1 = [float(log(RelNs1[i],10)) for i in range(len(sams))]
            RelNs3 = [abs(val_dictCF[p][9][i][3]/Mzx0) for i in range(len(sams))]
            Evals3 = [float(log(RelNs3[i],10)) for i in range(len(sams))]
        elif FEM[a] == "OSF":
            RelNs1 = [abs(val_dictOSF[p][9][i][1]/Mzx0) for i in range(len(sams))]
            Evals1 = [float(log(RelNs1[i],10)) for i in range(len(sams))]
            RelNs3 = [abs(val_dictOSF[p][9][i][3]/Mzx0) for i in range(len(sams))]
            Evals3 = [float(log(RelNs3[i],10)) for i in range(len(sams))]
        Nvals1,Nvals3 = [[],[]]
        xvals1 = np.linspace(min(xvals2),max(xvals2),100)
        LS_coef1 = np.polyfit(xvals2,Evals1,1)
        for i in range(len(xvals1)):
            Nvals1.append(sum([LS_coef1[k]*(xvals1[i]**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]))
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.plot(xvals1,Nvals1,color=col,label="{} DD ROC,(p,B) = ({},{})".format(FEM[a],p,round(abs(LS_coef1[0]),3)))
        # plt.plot(xvals2,Evals1,color=col,linestyle="dashed")
        plt.plot(xvals2,[sum([LS_coef1[k]*(i**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]) for i in xvals2],color=col,marker="o")

        LS_coef1 = np.polyfit(xvals2,Evals3,1)
        for i in range(len(xvals1)):
            Nvals3.append(sum([LS_coef1[k]*(xvals1[i]**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]))
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.plot(xvals1,Nvals3,color=col,label="{} WE ROC,(p,B) = ({},{})".format(FEM[a],p,round(abs(LS_coef1[0]),3)))
        # plt.plot(xvals2,Evals3,color=col,linestyle="dashed")
        plt.plot(xvals2,[sum([LS_coef1[k]*(i**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]) for i in xvals2],color=col,marker="o")
plt.xlabel("-log10(dx)")
plt.ylabel("log10(e_abs)")
plt.title("Rate of Convergence of Beam's Bending Moment at M(0) using p = {}".format(GL_points))
plt.grid()
plt.legend()
plt.show()
print()

# Shear Force
for a in range(len(FEM)):
    for p in GL_points:
        xvals2 = [float(-log(L/i,10)) for i in sams]
        #### Calculating Absolute Error using Direct Differentiation of the FE Solution
        if FEM[a] == "CF":
            RelNs2 = [abs(val_dictCF[p][9][i][0]/Vyx0) for i in range(len(sams))]
            Evals2 = [float(log(RelNs2[i],10)) for i in range(len(sams))]
            RelNs4 = [abs(val_dictCF[p][9][i][2]/Vyx0) for i in range(len(sams))]
            Evals4 = [float(log(RelNs4[i],10)) for i in range(len(sams))]
        elif FEM[a] == "OSF":
            RelNs2 = [abs(val_dictOSF[p][9][i][0]/Vyx0) for i in range(len(sams))]
            Evals2 = [float(log(RelNs2[i],10)) for i in range(len(sams))]
            RelNs4 = [abs(val_dictOSF[p][9][i][2]/Vyx0) for i in range(len(sams))]
            Evals4 = [float(log(RelNs4[i],10)) for i in range(len(sams))]
        Nvals2,Nvals4 = [[],[]]
        xvals1 = np.linspace(min(xvals2),max(xvals2),100)
        LS_coef1 = np.polyfit(xvals2,Evals2,1)
        for i in range(len(xvals1)):
            Nvals2.append(sum([LS_coef1[k]*(xvals1[i]**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]))
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.plot(xvals1,Nvals2,color=col,label="{} DD ROC,(p,B) = ({},{})".format(FEM[a],p,round(abs(LS_coef1[0]),3)))
        # plt.plot(xvals2,Evals2,color=col,linestyle="dashed")
        plt.plot(xvals2,[sum([LS_coef1[k]*(i**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]) for i in xvals2],color=col,marker="o")

        LS_coef1 = np.polyfit(xvals2,Evals4,1)
        for i in range(len(xvals1)):
            Nvals4.append(sum([LS_coef1[k]*(xvals1[i]**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]))
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.plot(xvals1,Nvals4,color=col,label="{} WE ROC,(p,B) = ({},{})".format(FEM[a],p,round(abs(LS_coef1[0]),3)))
        # plt.plot(xvals2,Evals4,color=col,linestyle="dashed") 
        plt.plot(xvals2,[sum([LS_coef1[k]*(i**j) for j,k in enumerate(reversed(range(len(LS_coef1))))]) for i in xvals2],color=col,marker="o")
plt.xlabel("-log10(dx)")
plt.ylabel("log10(e_abs)")
plt.title("Rate of Convergence of Beam's Shear Force at V(0) using p = {}".format(GL_points))
plt.grid()
plt.legend()
plt.show()
print()

#### Data to CSV section
tempd = val_dictCF
for p in [4]:
    if p in GL_points:
        sam_m = max(sams)
        file_str = "n = {},v(x),dv/dx(x),Mz(x),Vy(x)\n".format(sam_m)
        for i in range(1,sam_m+2):
            temp = "{}x/{}L,{},{},{},{}\n".format(i-1,sam_m,tempd[p][5][-1][i-1],tempd[p][6][-1][i-1],tempd[p][7][-1][i-1],tempd[p][8][-1][i-1])
            file_str += temp
        file = open("U_N charts p = {}.csv".format(p),"w")
        file.write(file_str)
        file.close()
print()