## Libary
import numpy as np
import matplotlib.pyplot as plt

## Input Parameter
M0 = 1.725
M = np.array([0, 0.5, 1, 1.5, M0, 2])
Mplot = np.arange(0,2,0.005)
h = 42000*1000# Jk/kg
y = 1.4
cp = 1004 # J/kg*K
Alt = 60000 # ft
T0 =  216.7 # K Assume standard air condition at 37000 ft altitude
Tt4 = 1600 # Kelvin
pic = np.arange(2,1000,1)
TS = np.zeros(len(M),dtype = object)
S = np.zeros(len(M),dtype = object)
f = np.zeros(len(M),dtype = object)
etaT = np.zeros(len(M),dtype = object)
etaP = np.zeros(len(M),dtype = object)
etaO = np.zeros(len(M),dtype = object)

## Calculation
R = (y - 1)*cp/y
a0 = np.sqrt(y*R*T0)
tl = Tt4/T0

def tccal(pic):
    return pic**((y-1)/y)
def fcal(tc,tr):
    return cp*T0*(tl-(tr*tc))/h
def trcal(M):
    return 1 + (y-1)*(M**2)/2
def Scal(f,TS):
    return (f/TS)*10**6
def v9cal(tr,tc,tt):
    return a0*np.sqrt(2*tl*((tr*tc*tt)-1)/((y-1)*tr*tc))
def TScal(v9,M):
    return (v9 - (M*a0)) 
def ttcal(tr,tc,tl):
    return 1 - tr*(tc-1)/tl
def etaTcal(tr,tc):
    return 1 - 1/(tr*tc)
def etaPcal(M,v9,a0):
    return 2*M/((v9/a0)+M)
def etaOcal(etaT,etaP):
    return etaT*etaP

tc = tccal(pic)
for i in range(len(M)):
    tr = trcal(M[i])
    tt = ttcal(tr,tc,tl)
    v9 = v9cal(tr,tc,tt)
    f[i] = fcal(tc,tr)
    TS[i] = TScal(v9,M[i]) # Specific thrust
    S[i] = Scal(f[i],TS[i]) # Thrust Specific Fuel Consumption
    etaT[i] = etaTcal(tr,tc) # Thermal efficiency
    etaP[i] = etaPcal(M[i],v9,a0) # Propulsive efficiency
    etaO[i] = etaOcal(etaT[i],etaP[i]) # Overall efficiency


## Optimization
MAXTS = np.max(TS[4])
MAXDT = MAXTS/a0
MAXpic = pic[np.argmax(TS[4])]
MAXtc = tccal(MAXpic)
MAXtr = trcal(M0)
MAXtt = ttcal(MAXtr,MAXtc,tl)
MAXf = fcal(MAXtc,MAXtr)
MAXS = MAXf/MAXTS
MAXetaTH = etaTcal(MAXtr,MAXtc)
MAXv9 = v9cal(MAXtr,MAXtc,MAXtt)
MAXetaP = etaPcal(M0,MAXv9,a0)
MAXetaO = MAXetaTH * MAXetaP
print("Maximum Dimensionless Thrust is: %1f at compressor ratio of %1f" % (MAXDT,MAXpic))
print("Fuel to Air ratio at maximum dimensionless thrust is %1f at compresor ratio of %1f" % (MAXf,MAXpic))
print("Thrust Speficic Fuel Consumption at maximum dimensionless thrust is %1f at compresor ratio of %1f" % (MAXS,MAXpic))
print("Thermal efficiency at maximum dimensionless thrust is %1f" % MAXetaTH)
print("Propulsive efficiency at maximum dimensionless thrust is %1f" % MAXetaP)
print("Overall efficiency at maximum dimensionless thrust is %1f" % MAXetaO)

## Vary M for 0 to 2
tcv = tccal(MAXpic)
trv = trcal(Mplot)
ttv = ttcal(trv,tcv,tl)
v9v = v9cal(trv,tcv,ttv)
TSv = TScal(v9v,Mplot)
fv = fcal(tcv,trv)
Sv = Scal(fv,TSv)
etaTv = etaTcal(trv,tcv)
etaPv = etaPcal(Mplot,v9v,a0)
etaOv = etaOcal(etaTv,etaPv)

## Plot for pic = different value

tcp2 = tccal(2)
ttp2 = ttcal(trv,tcp2,tl)
v9p2 = v9cal(trv,tcp2,ttp2)
TSp2 = TScal(v9p2,Mplot)
fp2 = fcal(tcp2,trv)
Sp2 = Scal(fp2,TSp2)
etaTp2 = etaTcal(trv,tcp2)
etaPp2 = etaPcal(Mplot,v9p2,a0)
etaOp2 = etaOcal(etaTp2,etaPp2)

tcp10 = tccal(10)
ttp10 = ttcal(trv,tcp10,tl)
v9p10 = v9cal(trv,tcp10,ttp10)
TSp10 = TScal(v9p10,Mplot)
fp10 = fcal(tcp10,trv)
Sp10 = Scal(fp10,TSp10)
etaTp10 = etaTcal(trv,tcp10)
etaPp10 = etaPcal(Mplot,v9p10,a0)
etaOp10 = etaOcal(etaTp10,etaPp10)

tcp20 = tccal(20)
ttp20 = ttcal(trv,tcp20,tl)
v9p20 = v9cal(trv,tcp20,ttp20)
TSp20 = TScal(v9p20,Mplot)
fp20 = fcal(tcp20,trv)
Sp20 = Scal(fp20,TSp20)
etaTp20 = etaTcal(trv,tcp20)
etaPp20 = etaPcal(Mplot,v9p20,a0)
etaOp20 = etaOcal(etaTp20,etaPp20)

## Plot
lw = 1.8 # design point linewidth
plt.figure(1,figsize=(8,6))
for i in range(len(M)):
    plt.title("Dimensionless Thrust at various Compressor Pressure Ratio")
    #if i == 4:
    plt.plot(pic,TS[4]/a0, linestyle ='solid', linewidth = lw, color = 'black')
    #else:
        #plt.plot(pic,TS[i]/a0, linestyle ='dashed')
    plt.legend(["M = 1.725"])
    plt.ylabel("Dimensionless Thrust [-]")
    plt.xlabel("Compressor Ratio $\pi_c$ [-]" )
    plt.savefig("dimTatpic.png")
    
plt.figure(2,figsize=(8,6))
for i in range(len(M)):
    plt.title("Thrust Specific Fuel Consumption at various Compressor Pressure Ratio")
    #if i == 4:
    plt.plot(pic,S[4], linestyle ='solid', linewidth = lw, color = 'black')
    #else:
        #plt.plot(pic,S[i], linestyle ='dashed')
    plt.legend(["M = 1.725"])
    plt.ylabel("Thrust Specific Fuel Consumption TSFC [mg/N-s]" )
    plt.xlabel("Compressor Ratio $\pi_c$ [-]" )

plt.figure(3,figsize=(8,6))
for i in range(len(M)):
    plt.title("Fuel to Air Ratio at various Compressor Pressure Ratio")
    #if i == 4:
    plt.plot(pic,f[4], linestyle ='solid', linewidth = lw, color = 'black')
    #else:
        #plt.plot(pic,f[i], linestyle ='dashed')
    plt.legend(["M = 1.725"])
    plt.ylabel("Fuel to Air Ratio [-]")
    plt.xlabel("Compressor Ratio $\pi_c$ [-]" )

plt.figure(4,figsize=(8,6))
for i in range(len(M)):
    plt.title("Thermal Efficiency at various Compressor Pressure Ratio")
    #if i == 4:
    plt.plot(pic,etaT[4], linestyle ='solid', linewidth = lw, color = 'black')
    #else:
        #plt.plot(pic,etaT[i], linestyle ='dashed')
    plt.legend(["M = 1.725"])
    plt.ylabel("Thermal Efficiency [-]")
    plt.xlabel("Compressor Ratio $\pi_c$ [-]" )

plt.figure(5,figsize=(8,6))
for i in range(len(M)):
    plt.title("Propulsive Efficiency at various Compressor Pressure Ratio")
    #if i == 4:
    plt.plot(pic,etaP[4], linestyle ='solid', linewidth = lw, color = 'black')
    #else:
    #plt.plot(pic,etaP[i], linestyle ='dashed')
    plt.legend(["M = 1.725"])
    plt.ylabel("Propulsive Efficiency [-]")
    plt.xlabel("Compressor Ratio $\pi_c$ [-]" )

plt.figure(6,figsize=(8,6))
for i in range(len(M)):
    plt.title("Overall Efficiency")
    #if i == 4:
    plt.plot(pic,etaO[4], linestyle ='solid', linewidth = lw, color = 'black')
    #else:
        #plt.plot(pic,etaO[i], linestyle ='dashed')
    plt.legend(["M = 1.725"])
    plt.ylabel("Overall Efficiency [-]")
    plt.xlabel("Compressor Ratio $\pi_c$ [-]" )





plt.figure(7,figsize=(8,6))
plt.title("Dimensionless Thrust at Compressor Ratio of 6.45 at Mach Number 0 to 2")
plt.plot(Mplot,TSv/a0,color = 'black',label='$\pi_c$ = 6.45')
#plt.plot(Mplot,TSp2/a0, linestyle = 'dashed')
#plt.plot(Mplot,TSp10/a0, linestyle = 'dashed')
#plt.plot(Mplot,TSp20/a0, linestyle = 'dashed')
plt.legend()
plt.ylabel("Dimensionless Thrust [-]")
plt.xlabel("Inlet Mach Number [-]" )

plt.figure(8,figsize=(8,6))
plt.plot(Mplot,Sv,color = 'black',label='$\pi_c$ = 6.45')
#plt.plot(Mplot,Sp2, linestyle = 'dashed')
#plt.plot(Mplot,Sp10, linestyle = 'dashed')
#plt.plot(Mplot,Sp20, linestyle = 'dashed')
plt.legend()
plt.title("Thrust Specific Fuel Consumption at Compressor Ratio of 6.45 at Mach Number 0 to 2")
plt.ylabel("Thrust Specific Fuel Consumption [mg/N-s]")
plt.xlabel("Inlet Mach Number [-]" )

plt.figure(9,figsize=(8,6))
plt.plot(Mplot,fv,color = 'black',label='$\pi_c$ = 6.45')
#plt.plot(Mplot,fp2, linestyle = 'dashed')
#plt.plot(Mplot,fp10, linestyle = 'dashed')
#plt.plot(Mplot,fp20, linestyle = 'dashed')
plt.legend()
plt.title("Fuel to Air Ratio at Compressor Ratio of 6.45 at Mach Number 0 to 2")
plt.ylabel("Fuel to Air Ratio [-]")
plt.xlabel("Inlet Mach Number [-]" )

plt.figure(10,figsize=(8,6))
plt.title("Thermal Efficiency at Compressor Ratio of 6.45 at Mach Number 0 to 2")
plt.plot(Mplot,etaTv,color = 'black', label='$\pi_c$ = 6.45')
plt.ylabel("Thermal Efficiency [-]")
plt.xlabel("Inlet Mach Number [-]" )
plt.legend()

plt.figure(11,figsize=(8,6))
plt.title("Propulsive Efficiency at Compressor Ratio of 6.45 at Mach Number 0 to 2")
plt.plot(Mplot,etaPv,color = 'black',label='$\pi_c$ = 6.45')
plt.ylabel("Propulsive Efficiency [-]")
plt.xlabel("Inlet Mach Number [-]" )
plt.legend()

plt.figure(12,figsize=(8,6))
plt.plot(Mplot,etaOv,color = 'black',label='$\pi_c$ = 6.45')
plt.title("Overall Efficiency at Compressor Ratio of 6.45 at Mach Number 0 to 2")
plt.ylabel("Overall Efficiency [-]")
plt.xlabel("Inlet Mach Number [-]" )
plt.legend()
#plt.plot(Mplot,etaOp2, linestyle = 'dashed')
#plt.plot(Mplot,etaOp10, linestyle = 'dashed')
#plt.plot(Mplot,etaOp20, linestyle = 'dashed')
plt.show()