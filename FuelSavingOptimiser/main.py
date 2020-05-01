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


BPlot = False
nan = float('nan')

# import CSV files with the following data: "Time","Ground Speed","G Force Lat","G Force Long","ThrottleRaw","BrakeRaw","FuelUsePerHour","Fuel Density","Lap Distance","FuelLevel","LapDistPct","aTrackIncline","GPS Altitude"
filename ='fordgt2017_monza full 2020-04-04 15-09-18_Stint_1.csv'
header = ["tLap","vCar","gLat","gLong","rThrottle","rBrake","QFuel","rhoFuel","sLap","FuelLevel","LapDistPct","aTrackIncline","GPSAltitude"]

# declare baseline data dict and fill it from CSV
d = {}

for i in range(0,len(header)):
    d[header[i]] = np.array([])

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        for i in range(0, len(header)):
            d[header[i]] = np.append(d[header[i]], float(line[i]))

d['sLap'] = d['sLap'] - d['sLap'][0]  # correct distance data

# do some calculations for baseline data
d['dt'] = np.diff(d['tLap'])
d['ds'] = np.diff(d['sLap'])
d = calcFuel(d)
d = calcLapTime(d)

# find apex points
NApex = scipy.signal.find_peaks(200-d['vCar'], height=10, prominence=1)
NApex = NApex[0]
dNApex = np.diff(d['sLap'][NApex]) < 50
if any(dNApex):
    for i in range(0, len(dNApex)):
        if dNApex[i]:
            NApex = np.delete(NApex, i+1)

plt.figure()
plt.plot(d['sLap'], d['vCar'], 'k', label='Speed')
plt.grid()
plt.title('Apex Points')
plt.xlabel('sLap [m]')
plt.ylabel('vCar [m/s]')
plt.scatter(d['sLap'][NApex], d['vCar'][NApex], label='Apex Points')
plt.legend()
plt.show(block=False)

# cut lap at first apex
sLapCut = d['sLap'][NApex[0]]
tLapCut = d['tLap'][NApex[0]]

# create new data dict for cut lap --> c
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

# re-do calculations for cut lap
d['dt'] = np.diff(d['tLap'])
d['ds'] = np.diff(d['sLap'])
d = calcFuel(d)
d = calcLapTime(d)
NApex = NApex[1:len(NApex)] - NApex[0]
NApex = np.append(NApex, len(d['tLap'])-1)
d['rBrake'][0] = 0.1  # fudging around to find the first brake point

# find potential lift point (from full throttle to braking)
NWOT = scipy.signal.find_peaks(d['rThrottle'], height=100, plateau_size=80)
NBrake = scipy.signal.find_peaks(100-d['rBrake'], height=100, plateau_size=40)

if not len(NWOT[1]) == len(NBrake[1]):
    print('Error! Number of brake application and full throttle point don nit match!')
    quit()

# sections for potential lifting
NWOT = NWOT[1]['left_edges']
NBrake = NBrake[1]['right_edges']

plt.figure()
plt.plot(d['sLap'], d['vCar'], label='Speed - Push Lap')
plt.scatter(d['sLap'][NWOT], d['vCar'][NWOT], label='Full Throttle Points')
plt.scatter(d['sLap'][NBrake], d['vCar'][NBrake], label='Brake Points')
plt.scatter(d['sLap'][NApex], d['vCar'][NApex], label='Apex Points')
plt.grid()
plt.legend()
plt.title('Sections')
plt.xlabel('sLap [m]')
plt.ylabel('vCar [m/s]')
plt.show(block=False)

print('\nPush Lap:')
print('LapTime :', np.round(d['tLap'][-1], 3))
print('VFuel :', np.round(d['VFuel'][-1], 3))

# Find earliest lift points. Assumption: arriving at apex with apex speed but no brake application
NLiftEarliest = np.array([], dtype='int32')
d_temp = copy.deepcopy(d)
for i in range(0, len(NWOT)):
    d_temp, n = stepBwds(d_temp, NBrake[i] + int(0.85*(NApex[i]-NBrake[i])))
    NLiftEarliest = np.append(NLiftEarliest, n)

plt.figure()
plt.title('Earliest Lift Points')
plt.xlabel('sLap [m]')
plt.ylabel('vCar [m/s]')
plt.plot(d['sLap'], d['vCar'], label='Speed')
plt.plot(d_temp['sLap'], d_temp['vCar'], label='Maximum Lifting')
plt.scatter(d_temp['sLap'][NLiftEarliest], d_temp['vCar'][NLiftEarliest], label='Earliest Lift Point')
plt.grid()
plt.legend()
plt.show(block=False)

#  for each lifting zone calculate various ratios of lifting
rLift = np.linspace(0, 1, 50)

VFuelRLift = np.zeros((len(NLiftEarliest), len(rLift)))
tLapRLift = np.zeros((len(NLiftEarliest), len(rLift)))

for i in range(0, len(NLiftEarliest)):
    for k in range(1, len(rLift)):
        tLapRLift[i, k], VFuelRLift[i, k], R = costFcn([rLift[k]], copy.deepcopy(d), [NLiftEarliest[i]], [NBrake[i]], None, False)

# get fuel consumption and lap time differences compared to original lap
tLapRLift = tLapRLift - d['tLap'][-1]
VFuelRLift = VFuelRLift - d['VFuel'][-1]

# remove outliners
VFuelRLift[tLapRLift == 0] = nan
tLapRLift[tLapRLift == 0] = nan
tLapRLift[:, 0] = 0
VFuelRLift[:, 0] = 0

# fit lap time and fuel consumption polynomial for each lifting zone
tLapPolyFit = np.zeros((len(NLiftEarliest), 6))
VFuelPolyFit = np.zeros((len(NLiftEarliest), 6))

rLiftPlot = np.linspace(0, 1, 1000)

for i in range(0, len(NLiftEarliest)):
    # remove nan indices
    yTime = tLapRLift[i, :]
    yFuel = VFuelRLift[i, :]
    x = rLift[~np.isnan(yTime)]
    f = yFuel[~np.isnan(yTime)]
    t = yTime[~np.isnan(yTime)]

    tLapPolyFit[i, :], temp = scipy.optimize.curve_fit(poly6, x, t)
    VFuelPolyFit[i, :],  temp = scipy.optimize.curve_fit(poly6, x, f)

    if BPlot:
        plt.figure()
        plt.title('Lap Time Loss - Lift Zone ' + str(i+1))
        plt.xlabel('rLift [-]')
        plt.ylabel('dtLap [s]')
        plt.scatter(rLift, tLapRLift[i, :])
        plt.plot(rLiftPlot, poly6(rLiftPlot, tLapPolyFit[i, 0], tLapPolyFit[i, 1], tLapPolyFit[i, 2], tLapPolyFit[i, 3], tLapPolyFit[i, 4], tLapPolyFit[i, 5]))
        plt.grid()
        plt.show(block=False)

        plt.figure()
        plt.title('Fuel Save - Lift Zone ' + str(i+1))
        plt.xlabel('rLift [-]')
        plt.ylabel('dVFuel [l]')
        plt.scatter(rLift, VFuelRLift[i, :])
        plt.plot(rLiftPlot, poly6(rLiftPlot, VFuelPolyFit[i, 0], VFuelPolyFit[i, 1], VFuelPolyFit[i, 2], VFuelPolyFit[i, 3], VFuelPolyFit[i, 4], VFuelPolyFit[i, 5]))
        plt.grid()
        plt.show(block=False)

# maximum lift
tLapMaxSave, VFuelMaxSave, R = costFcn(np.ones(len(NLiftEarliest)), d, NLiftEarliest, NBrake, None, False)

# optimisation for 100 steps between maximum lift and push
VFuelTGT = np.max([3.1, VFuelMaxSave])
VFuelTGT = np.linspace(VFuelMaxSave, d['VFuel'][-1], 100)

print('\nMaximum Lift:')
print('LapTime :', np.round(tLapMaxSave, 3))
print('VFuel :', np.round(VFuelMaxSave, 3))

# bounds and constaints
bounds = [(0, 1)]*6
LiftPointsVsFuelCons = {'VFuelTGT': np.empty((len(VFuelTGT), 1)), 'LiftPoints': np.empty((len(VFuelTGT), len(NLiftEarliest)))}

result = []
fun = []

for i in range(0, len(VFuelTGT)):  # optimisation loop

    VFuelConsTGT = VFuelTGT[i] - d['VFuel'][-1]

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
plt.show(block=False)

plt.figure()
plt.title('rLift vs VFuelTGT')
plt.xlabel('VFuelTGT [l]')
plt.ylabel('rLift [-]')
for k in range(0, len(NLiftEarliest)):
    plt.plot(LiftPointsVsFuelCons['VFuelTGT'], LiftPointsVsFuelCons['LiftPoints'][:, k], label='Lift Zone ' + str(k+1))

plt.legend()
plt.grid()
plt.show(block=False)

# get LapDistPct

# transform back to original lap

# export data

print('Done')