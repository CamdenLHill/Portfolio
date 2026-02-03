# Camden Hill
# Vibrations 3DOF Spring-Mass Graphs

import matplotlib.pyplot as plt
from numpy import linspace, cos, sin, random #  log10,
samples = 201
colors = [(random.random(), random.random(), random.random()) for i in range(3)]
tvals = linspace(0,10,samples)
# qvals1_1 = [ for i in tvals]
qvals1 = [(5290551974883943321857555368343*cos((4194683690460669*i)/2251799813685248))/81129638414606681695789005144064 + (11535296628531247344761058440847*cos((8066794699710947*i)/9007199254740992))/10141204801825835211973625643008 - (1027705411782952230624116784481*cos((168717106719797*i)/2251799813685248))/5070602400912917605986812821504 for i in tvals]
qvals2 = [(19269644534413837997471175731061*cos((8066794699710947*i)/9007199254740992))/40564819207303340847894502572032 - (18259044287639583118635387863163*cos((4194683690460669*i)/2251799813685248))/81129638414606681695789005144064 - (7605632998561429321432730936177*cos((168717106719797*i)/2251799813685248))/10141204801825835211973625643008 for i in tvals]
qvals3 = [(3979356628048445145148084991439*cos((4194683690460669*i)/2251799813685248))/81129638414606681695789005144064 - (11423906815269411870917139027279*cos((8066794699710947*i)/9007199254740992))/40564819207303340847894502572032 - (153295134951245373342537373705*cos((168717106719797*i)/2251799813685248))/158456325028528675187087900672 for i in tvals]
qvals1_1 = [(5122518291258589*cos((8066794699710947*i)/9007199254740992))/4503599627370496 - (326407590323095*sin((8066794699710947*i)/9007199254740992))/36028797018963968 - (1793964180071981*cos((168717106719797*i)/2251799813685248))/9007199254740992 - (763524561562005*sin((168717106719797*i)/2251799813685248))/144115188075855872 for i in tvals]
qvals2_1 = [(4279980135576467*cos((8066794699710947*i)/9007199254740992))/9007199254740992 + (2253031697193365*sin((8066794699710947*i)/9007199254740992))/72057594037927936 - (3432122074237037*cos((168717106719797*i)/2251799813685248))/4503599627370496 + (658779502141065*sin((168717106719797*i)/2251799813685248))/36028797018963968 for i in tvals]
qvals3_1 = [- (1268445687830239*cos((8066794699710947*i)/9007199254740992))/4503599627370496 - (491023323904001*sin((8066794699710947*i)/9007199254740992))/72057594037927936 - (4345022842873827*cos((168717106719797*i)/2251799813685248))/4503599627370496 - (287147403442257*sin((168717106719797*i)/2251799813685248))/72057594037927936 for i in tvals]
qvals1_2 = [(23828045697879*sin((168717106719797*i)/2251799813685248))/1125899906842624 - (5457349067301733*cos((168717106719797*i)/2251799813685248))/36028797018963968 for i in tvals]
qvals2_2 = [(287616498463813*sin((168717106719797*i)/2251799813685248))/9007199254740992 - (3312039797527137*cos((168717106719797*i)/2251799813685248))/4503599627370496 for i in tvals]
qvals3_2 = [- (8812111525065365*cos((168717106719797*i)/2251799813685248))/9007199254740992 - (201893250796733*sin((168717106719797*i)/2251799813685248))/18014398509481984 for i in tvals]
qvals1_3 = [cos((168717106719797*i)/2251799813685248) for i in tvals]
qvals2_3 = [-0.5*cos((168717106719797*i)/2251799813685248) for i in tvals]
qvals3_3 = [-1.2*cos((168717106719797*i)/2251799813685248) for i in tvals]

plt.plot(tvals,qvals1,color="red",label="q1(t)")
plt.plot(tvals,qvals2,color="blue",label="q2(t)")
plt.plot(tvals,qvals3,color="green",label="q3(t)")
plt.grid()
plt.legend()
plt.xlabel("Time (sec)")
plt.ylabel("Relative Displacement (in)")
plt.title("Relative Displacement Of The 3DOF System")
plt.show()
print()

plt.plot(tvals,qvals1,color="red",label="q1(t)")
plt.plot(tvals,qvals2,color="blue",label="q2(t)")
plt.plot(tvals,qvals3,color="green",label="q3(t)")
plt.plot(tvals,qvals1_1,color=colors[0],linestyle="--",label="2-mode lsq q1(t)")
plt.plot(tvals,qvals2_1,color=colors[1],linestyle="--",label="2-mode lsq q2(t)")
plt.plot(tvals,qvals3_1,color=colors[2],linestyle="--",label="2-mode lsq q3(t)")
plt.grid()
plt.legend()
plt.xlabel("Time (sec)")
plt.ylabel("Relative Displacement (in)")
plt.title("Relative Displacement Of The 3DOF System With Least-Squares Method")
plt.show()
print()

plt.plot(tvals,qvals1,color="red",label="q1(t)")
plt.plot(tvals,qvals2,color="blue",label="q2(t)")
plt.plot(tvals,qvals3,color="green",label="q3(t)")
plt.plot(tvals,qvals1_2,color=colors[0],linestyle="--",label="1-mode lsq q1(t)")
plt.plot(tvals,qvals2_2,color=colors[1],linestyle="--",label="1-mode lsq q2(t)")
plt.plot(tvals,qvals3_2,color=colors[2],linestyle="--",label="1-mode lsq q3(t)")
plt.plot(tvals,qvals1_3,color="red",linestyle="--",label="1-mode q1(t)")
plt.plot(tvals,qvals2_3,color="blue",linestyle="--",label="1-mode q2(t)")
plt.plot(tvals,qvals3_3,color="green",linestyle="--",label="1-mode q3(t)")
plt.grid()
plt.legend()
plt.xlabel("Time (sec)")
plt.ylabel("Relative Displacement (in)")
plt.title("Relative Displacement Of The 3DOF System With Least-Squares Method")
plt.show()
print()

# plt.plot(tvals,qvals1_log,color="red",label="q1(t)")
# plt.plot(tvals,qvals2_log,color="blue",label="q2(t)")
# plt.plot(tvals,qvals3_log,color="green",label="q3(t)")
# plt.grid()
# plt.legend()
# plt.xlabel("Time (sec)")
# plt.ylabel("log10(Relative Displacement)")
# plt.title("Relative Displacement Of The 3DOF System (Log10)")
# plt.show()
# print()