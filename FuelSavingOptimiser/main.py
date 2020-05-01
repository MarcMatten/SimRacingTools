import csv
import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt
import copy
from costFcn import costFcn


def calcFuel(x):
    x['mFuel'] = np.cumsum(np.append(0 ,x['dt'] * x['QFuel'][0:-1]))/1000
    x['VFuel'] = x['mFuel']/x['rhoFuel'][0]
    return x

def calcLapTime(x):
    x['tLap'] = np.cumsum(np.append(0, x['dt']))
    return x

def stepFwds(x, n):

    temp = copy.deepcopy(x)

    while temp['vCar'][n] < temp['vCar'][n+1]:
        vCar = temp['vCar'][n]
        gLong = getGLong(vCar, temp['aTrackIncline'][n])
        ds = temp['ds'][n]
        temp['dt'][n] = -vCar/gLong - np.sqrt( np.square(vCar / gLong) + 2*ds/gLong )
        temp['vCar'][n+1] = temp['vCar'][n] + gLong * temp['dt'][n]
        temp['QFuel'][n] = 0.61
        n=n+1

    return temp


def stepBwds(x, i):

    n = i
    temp = copy.deepcopy(x)

    while temp['vCar'][n] <= x['vCar'][n-1]:
        vCar = temp['vCar'][n]
        gLong = getGLong(vCar, temp['aTrackIncline'][n])
        ds = temp['ds'][n]
        temp['dt'][n] = -vCar/gLong - np.sqrt( np.square(vCar / gLong) + 2*ds/gLong )
        temp['vCar'][n-1] = temp['vCar'][n] - gLong * temp['dt'][n]
        temp['QFuel'][n] = 0.61
        n=n-1

    return temp, int(n)

def getGLong(v, aTrackIncline):
    return -0.00054902*np.square(v) +0.005411 * v - 1.0253 - np.sin(aTrackIncline/180*np.pi)

def poly6(x, a, b, c, d, e, f):
    return a + b * x + c * np.power(x, 2) + d * np.power(x, 3) + e * np.power(x, 4) + f * np.power(x, 5)

def objectiveLapTime(rLift, *args):
    tLapPolyFit = args[0]
    dt = 0
    for i in range(0,len(rLift)):
        dt = dt + poly6(rLift[i], tLapPolyFit[i, 0], tLapPolyFit[i, 1], tLapPolyFit[i, 2], tLapPolyFit[i, 3], tLapPolyFit[i, 4], tLapPolyFit[i, 5])
    return dt

def calcFuelConstraint(rLift, *args):
    VFuelPolyFit = args[0]
    VFuelConsTGT = args[1]
    dVFuel = 0
    for i in range(0,len(rLift)):
        dVFuel = dVFuel + poly6(rLift[i], VFuelPolyFit[i, 0], VFuelPolyFit[i, 1], VFuelPolyFit[i, 2], VFuelPolyFit[i, 3], VFuelPolyFit[i, 4], VFuelPolyFit[i, 5])
    return dVFuel - VFuelConsTGT

# def costFcn2(rLift, *args):
#     C = args[0]
#     NLiftEarliest = args[1]
#     NBrake = args[2]
#     for i in range(0, len(NLiftEarliest)):
#         C = stepFwds(C, int(NLiftEarliest[i] + (1 - rLift[i]) * (NBrake[i] - NLiftEarliest[i])))
#
#     C = calcFuel(C)
#     C = calcLapTime(C)
#
#     return C['tLap'][-1], C['VFuel'][-1], C

# import csv file
# filename ='porsche911cup_monza full 2019-08-04 17-49-13_Stint_1.csv'
# header = ["Time","Ground Speed","G Force Lat","G Force Long","ThrottleRaw","BrakeRaw","FuelUsePerHour","Fuel Density","Lap Distance"]
# header = ["tLap","vCar","gLat","gLong","rThrottle","rBrake","QFuel","rhoFuel","sLap"]

filename ='fordgt2017_monza full 2020-04-04 15-09-18_Stint_1.csv'
# header = ["Time","Ground Speed","G Force Lat","G Force Long","ThrottleRaw","BrakeRaw","FuelUsePerHour","Fuel Density","Lap Distance","FuelLevel","LapDistPct","aTrackIncline","GPS Altitude"]
header = ["tLap","vCar","gLat","gLong","rThrottle","rBrake","QFuel","rhoFuel","sLap","FuelLevel","LapDistPct","aTrackIncline","GPSAltitude"]

# declare data struct
d = {}

BPlot = True
nan = float('nan')

for i in range(0,len(header)):
    d[header[i]] = np.array([])

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        for i in range(0, len(header)):
            d[header[i]] = np.append(d[header[i]], float(line[i]))

d['sLap'] = d['sLap'] - d['sLap'][0]
d['dt'] = np.diff(d['tLap'])
d['ds'] = np.diff(d['sLap'])
d = calcFuel(d)
d = calcLapTime(d)

# find first corner
if BPlot:
    plt.figure()
    plt.plot(d['sLap'], d['vCar'], 'k')
    plt.grid()
    plt.title('Apex Points')
    plt.xlabel('sLap [m]')
    plt.ylabel('vCar [m/s]')
NApex = scipy.signal.find_peaks(200-d['vCar'], height=10, prominence=1)
NApex = NApex[0]
dNApex  = np.diff(d['sLap'][NApex]) < 50
if any(dNApex):
    for i in range(0, len(dNApex)):
        if dNApex[i]:
            NApex = np.delete(NApex,i+1)

if BPlot:
    plt.scatter(d['sLap'][NApex], d['vCar'][NApex])
    # plt.show(block=False)

# cut lap at first corner
sLapCut = d['sLap'][NApex[0]]
tLapCut = d['tLap'][NApex[0]]

temp = copy.deepcopy(d)
d = {}
keys = list(temp.keys())

for i in range(0,len(temp)):
    if keys[i] == 'tLap':
        d[keys[i]] = temp[keys[i]][NApex[0]:-1] - tLapCut
        d[keys[i]] = np.append(d[keys[i]], temp[keys[i]][0:NApex[0]] + d[keys[i]][-1] + temp['dt'][-1])
    elif keys[i] == 'sLap':
        d[keys[i]] = temp[keys[i]][NApex[0]:-1] - sLapCut
        d[keys[i]] = np.append(d[keys[i]], temp[keys[i]][0:NApex[0]] + d[keys[i]][-1] + temp['ds'][-1])
    else:
        d[keys[i]] = temp[keys[i]][NApex[0]:-1]
        d[keys[i]] = np.append(d[keys[i]], temp[keys[i]][0:NApex[0]])

d['dt'] = np.diff(d['tLap'])
d['ds'] = np.diff(d['sLap'])
d = calcFuel(d)
d = calcLapTime(d)
NApex = NApex[1:len(NApex)] - NApex[0]
NApex = np.append(NApex, len(d['tLap'])-1)
d['rBrake'][0] = 0.1 # fudging around to find the first brake point

# find potential lift point (from full throttle to braking)
NWOT = scipy.signal.find_peaks(d['rThrottle'], height=100, plateau_size=80)
# plt.plot(d['sLap'], d['rThrottle'])
# plt.grid()
# plt.scatter(d['sLap'][NWOT[1]['left_edges']], d['rThrottle'][NWOT[1]['left_edges']])
# plt.show()

NBrake = scipy.signal.find_peaks(100-d['rBrake'], height=100, plateau_size=40)
# plt.plot(d['sLap'], d['rBrake'])
# plt.grid()
# plt.scatter(d['sLap'][NBrake[1]['right_edges']], d['rBrake'][NBrake[1]['right_edges']])

if not len(NWOT[1]) == len(NBrake[1]):
    print('Error! Number of brake application and full throttle point don nit match!')
    quit()

# sections for lifting
NWOT = NWOT[1]['left_edges']
NBrake = NBrake[1]['right_edges']

if BPlot:
    plt.figure()
    plt.plot(d['sLap'], d['vCar'])
    plt.scatter(d['sLap'][NWOT], d['vCar'][NWOT])
    plt.scatter(d['sLap'][NBrake], d['vCar'][NBrake])
    plt.scatter(d['sLap'][NApex], d['vCar'][NApex])
    plt.grid()
    plt.title('Sections')
    plt.xlabel('sLap [m]')
    plt.ylabel('vCar [m/s]')
    # plt.show(block=False)

print('Initial Values (Push):')
print('LapTime :', d['tLap'][-1])
print('VFuel :', d['VFuel'][-1])

# recalc lifting phase
# d2 = stepFwds(d, 750)

NLiftEarliest = np.array([], dtype='int32')
d_temp = copy.deepcopy(d)
for i in range(0, len(NWOT)):
    d_temp, n = stepBwds(d_temp, NBrake[i] + int(0.85*(NApex[i]-NBrake[i])))
    NLiftEarliest = np.append(NLiftEarliest, n)

if BPlot:
    plt.figure()
    plt.title('Lift Points')
    plt.xlabel('sLap [m]')
    plt.ylabel('vCar [m/s]')
    plt.plot(d['sLap'], d['vCar'])
    plt.plot(d_temp['sLap'], d_temp['vCar'])
    plt.scatter(d_temp['sLap'][NLiftEarliest], d_temp['vCar'][NLiftEarliest])
    plt.grid()
    plt.show(block=False)

#  for each section calc 10%, 20%, 30%, ... lifting
r = np.linspace(0, 1, 50)

VFuelMatrix = np.zeros((len(NLiftEarliest), len(r)))
tLapMatrix = np.zeros((len(NLiftEarliest), len(r)))

for i in range(0, len(NLiftEarliest)):
    for k in range(1, len(r)):
        c = copy.deepcopy(d)
        tLapMatrix[i, k], VFuelMatrix[i, k], R = costFcn([r[k]], c, [NLiftEarliest[i]], [NBrake[i]], None, False)
        print(VFuelMatrix[i, k])

tLapMatrix = tLapMatrix - d['tLap'][-1]

VFuelMatrix = VFuelMatrix - d['VFuel'][-1]

# remove outliners
VFuelMatrix[tLapMatrix == 0] = nan
tLapMatrix[tLapMatrix == 0] = nan

tLapMatrix[:, 0] = 0
VFuelMatrix[:, 0] = 0

# fit
tLapPolyFit = np.zeros((len(NLiftEarliest), 6))
VFuelPolyFit = np.zeros((len(NLiftEarliest), 6))

r2 = np.linspace(0, 1, 1000)

for i in range(0, len(NLiftEarliest)):
    # remove nan indices
    tLapValues = tLapMatrix[i, :]
    VFuelValues = VFuelMatrix[i, :]
    R = r[~np.isnan(tLapValues)]
    VFuelValues = VFuelValues[~np.isnan(tLapValues)]
    tLapValues = tLapValues[~np.isnan(tLapValues)]

    tLapPolyFit[i, :], temp = scipy.optimize.curve_fit(poly6, R, tLapValues)
    VFuelPolyFit[i, :],  temp = scipy.optimize.curve_fit(poly6, R, VFuelValues)

    if BPlot:
        plt.figure()
        plt.title('Lap Time Loss - Lift Zone ' + str(i+1))
        plt.xlabel('rLift [-]')
        plt.ylabel('dtLap [s]')
        plt.scatter(r, tLapMatrix[i, :])
        plt.plot(r2, poly6(r2, tLapPolyFit[i, 0], tLapPolyFit[i, 1], tLapPolyFit[i, 2], tLapPolyFit[i, 3], tLapPolyFit[i, 4], tLapPolyFit[i, 5]))
        plt.grid()
        plt.show(block=False)

        plt.figure()
        plt.title('Fuel Save - Lift Zone ' + str(i+1))
        plt.xlabel('rLift [-]')
        plt.ylabel('dVFuel [l]')
        plt.scatter(r, VFuelMatrix[i, :])
        plt.plot(r2, poly6(r2, VFuelPolyFit[i, 0], VFuelPolyFit[i, 1], VFuelPolyFit[i, 2], VFuelPolyFit[i, 3], VFuelPolyFit[i, 4], VFuelPolyFit[i, 5]))
        plt.grid()
        plt.show(block=False)



#  optimisation
# c1 = copy.deepcopy(d)
# tLap1, VFuel1, c1 = costFcn([0.10756069,0.29391562,0.19019765,0.16411668,0.16653178,0.11581755], c1, NLiftEarliest, NBrake, None, False)
# c2 = copy.deepcopy(d)
# tLap2, VFuel2, c2 = costFcn([0.21527038,0.4561575,0.34679122,0.28843153,0.30255608,0.17479146], c2, NLiftEarliest, NBrake, None, False)
#
# plt.figure()
# plt.title('Optimisation Results')
# plt.xlabel('sLap [m]')
# plt.ylabel('vCar [m/s]')
# plt.plot(d['sLap'], d['vCar'], label='Push 105.860 s, 3.274 l')
# plt.plot(c1['sLap'], c1['vCar'], label='TGT 106.117 s, 3.050 l')
# plt.plot(c2['sLap'], c2['vCar'], label='TGT 106.595 s, 2.900 l')
# plt.legend()
# plt.grid()

# maximum saving
tLapMaxSave, VFuelMaxSave, R = costFcn(np.ones(len(NLiftEarliest)), d, NLiftEarliest, NBrake, None, False)

bounds = [(0, 1)]*6
VFuelTGT = np.max([3.1, VFuelMaxSave])

VFuelTGT = np.linspace(VFuelMaxSave, d['VFuel'][-1], 100)

LiftPointsVsFuelCons = {'VFuelTGT': np.empty((len(VFuelTGT), 1)), 'LiftPoints': np.empty((len(VFuelTGT), len(NLiftEarliest)))}

result = []
fun = []

for i in range(0, len(VFuelTGT)):

    VFuelConsTGT = VFuelTGT[i] - d['VFuel'][-1]

    # FuelConstraint = scipy.optimize.NonlinearConstraint(calcFuelConstraint, VFuelConsTGT, VFuelConsTGT)

    FuelConstraint = {'type': 'eq', 'fun': calcFuelConstraint, 'args': (VFuelPolyFit, VFuelConsTGT)}

    temp_result = scipy.optimize.minimize(objectiveLapTime, [0.0]*6, args=(tLapPolyFit, VFuelPolyFit), method='SLSQP', bounds=bounds, constraints=FuelConstraint, options={'maxiter': 10000, 'ftol': 1e-09, 'iprint': 1, 'disp': False})

    result.append(temp_result)
    fun.append(temp_result['fun'])

    LiftPointsVsFuelCons['VFuelTGT'][i] = VFuelTGT[i]
    LiftPointsVsFuelCons['LiftPoints'][i, :] = result[i]['x']

plt.figure()
plt.title('tLap vs VFuelTGT')
plt.xlabel('VFuelTGT [l]')
plt.ylabel('tLap [s]')
plt.plot(LiftPointsVsFuelCons['VFuelTGT'], fun)
plt.grid()
plt.show()

plt.figure()
plt.title('rLift vs VFuelTGT')
plt.xlabel('VFuelTGT [l]')
plt.ylabel('rLift [-]')
for k in range(0, len(NLiftEarliest)):
    plt.plot(LiftPointsVsFuelCons['VFuelTGT'], LiftPointsVsFuelCons['LiftPoints'][:, k], label='Lift Zone ' + str(k+1))

plt.legend()
plt.grid()
plt.show()

c = copy.deepcopy(d)
# bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]
# helpData = (c, NLiftEarliest, NBrake, 2.9, True)
# # cons = scipy.optimize.NonlinearConstraint(h, 0, 0)
# result = scipy.optimize.differential_evolution(costFcn, bounds, helpData, tol= 1e-9, maxiter=1000, popsize=60, disp=True) # , updating='deferred', workers=2, maxiter=2 ,popsize= 2
# # result = scipy.optimize.differential_evolution(costFcn, bounds, helpData, constraints=cons)
#
tLap, VFuel, R = costFcn(result['x'], c, NLiftEarliest, NBrake, None, False)

plt.figure()
plt.title('Optimisation Results')
plt.xlabel('sLap [m]')
plt.ylabel('vCar [m/s]')
plt.plot(d['sLap'], d['vCar'])
plt.plot(R['sLap'], R['vCar'])
plt.grid()

print('\nOptimisation Results:')
print('Results: ' + str(result['x']))
print('tLap: ', R['tLap'][-1], '(', str(d['tLap'][-1] + result.fun), ')')
print('VFuel: ', R['VFuel'][-1], '(', str(d['VFuel'][-1] + calcFuelConstraint(result['x'], VFuelPolyFit, 0)), ')')

plt.ioff()
plt.show(block=False)

print('Done')








