import csv
import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt
import copy
from costFcn import costFcn
import json


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

def saveJson(x):
    filepath ='FuelTGTLiftPoints.json'

    variables = list(x.keys())

    data = {}

    for i in range(0, len(variables)):
        data[variables[i]] = x[variables[i]].transpose().tolist()

    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

    print('Saved data ' + filepath)


BPlot = False
nan = float('nan')

# import CSV files with the following data: "Time", "Ground Speed", "G Force Lat", "G Force Long", "ThrottleRaw", "BrakeRaw", "FuelUsePerHour", "Fuel Density", "Lap Distance", "FuelLevel",
# "LapDistPct", "aTrackIncline", "GPS Altitude"
filename = 'fordgt2017_monza full 2020-04-04 15-09-18_Stint_1.csv'
header = ["tLap", "vCar", "gLat", "gLong", "rThrottle", "rBrake", "QFuel", "rhoFuel", "sLap", "FuelLevel", "LapDistPct", "aTrackIncline", "GPSAltitude"]

# declare baseline data dict and fill it from CSV
d = {}

for i in range(0, len(header)):
    d[header[i]] = np.array([])

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        for i in range(0, len(header)):
            d[header[i]] = np.append(d[header[i]], float(line[i]))

d['sLap'] = d['sLap'] - d['sLap'][0]  # correct distance data
# d['LapDistPct'] = d['LapDistPct'] - d['LapDistPct'][0]
# d['LapDistPct'] = d['LapDistPct'] / d['LapDistPct'][-1] *100

# do some calculations for baseline data
d['dt'] = np.diff(d['tLap'])
d['ds'] = np.diff(d['sLap'])
d['dLapDistPct'] = np.diff(d['LapDistPct'])
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
# create new data dict for cut lap --> c
temp = copy.deepcopy(d)
c = {}
keys = list(temp.keys())
NCut = len(temp[keys[i]][NApex[0]:-1])

for i in range(0, len(temp)):
    if keys[i] == 'tLap':
        c[keys[i]] = temp[keys[i]][NApex[0]:-1] - d['tLap'][NApex[0]]
        c[keys[i]] = np.append(c[keys[i]], temp[keys[i]][0:NApex[0]] + c[keys[i]][-1] + temp['dt'][-1])
    elif keys[i] == 'sLap':
        c[keys[i]] = temp[keys[i]][NApex[0]:-1] - d['sLap'][NApex[0]]
        c[keys[i]] = np.append(c[keys[i]], temp[keys[i]][0:NApex[0]] + c[keys[i]][-1] + temp['ds'][-1])
    elif keys[i] == 'LapDistPct':
        c[keys[i]] = temp[keys[i]][NApex[0]:-1] - d['LapDistPct'][NApex[0]]
        c[keys[i]] = np.append(c[keys[i]], temp[keys[i]][0:NApex[0]] + c[keys[i]][-1] + temp['dLapDistPct'][-1])
    else:
        c[keys[i]] = temp[keys[i]][NApex[0]:-1]
        c[keys[i]] = np.append(c[keys[i]], temp[keys[i]][0:NApex[0]])

# re-do calculations for cut lap
c['dt'] = np.diff(c['tLap'])
c['ds'] = np.diff(c['sLap'])
c['dLapDistPct'] = np.diff(c['LapDistPct'])
c = calcFuel(c)
c = calcLapTime(c)
NApex = NApex[1:len(NApex)] - NApex[0]
NApex = np.append(NApex, len(c['tLap'])-1) # fake apex for last index
c['rBrake'][0] = 0.1  # fudging around to find the first brake point

# find potential lift point (from full throttle to braking)
NWOT = scipy.signal.find_peaks(c['rThrottle'], height=100, plateau_size=80)
NBrake = scipy.signal.find_peaks(100-c['rBrake'], height=100, plateau_size=40)

if not len(NWOT[1]) == len(NBrake[1]):
    print('Error! Number of brake application and full throttle point don nit match!')
    quit()

# sections for potential lifting
NWOT = NWOT[1]['left_edges']
NBrake = NBrake[1]['right_edges']

plt.figure()
plt.plot(c['sLap'], c['vCar'], label='Speed - Push Lap')
plt.scatter(c['sLap'][NWOT], c['vCar'][NWOT], label='Full Throttle Points')
plt.scatter(c['sLap'][NBrake], c['vCar'][NBrake], label='Brake Points')
plt.scatter(c['sLap'][NApex], c['vCar'][NApex], label='Apex Points')

plt.grid()
plt.legend()
plt.title('Sections')
plt.xlabel('sLap [m]')
plt.ylabel('vCar [m/s]')
plt.show(block=False)

print('\nPush Lap:')
print('LapTime :', np.round(c['tLap'][-1], 3))
print('VFuel :', np.round(c['VFuel'][-1], 3))

# Find earliest lift points. Assumption: arriving at apex with apex speed but no brake application
NLiftEarliest = np.array([], dtype='int32')
c_temp = copy.deepcopy(c)
for i in range(0, len(NWOT)):
    c_temp, n = stepBwds(c_temp, NBrake[i] + int(0.85*(NApex[i]-NBrake[i])))
    NLiftEarliest = np.append(NLiftEarliest, n)

plt.figure()
plt.title('Earliest Lift Points')
plt.xlabel('sLap [m]')
plt.ylabel('vCar [m/s]')
plt.plot(c['sLap'], c['vCar'], label='Speed')
plt.plot(c_temp['sLap'], c_temp['vCar'], label='Maximum Lifting')
plt.scatter(c_temp['sLap'][NLiftEarliest], c_temp['vCar'][NLiftEarliest], label='Earliest Lift Point')
plt.grid()
plt.legend()
plt.show(block=False)

# for each lifting zone calculate various ratios of lifting
rLift = np.linspace(0, 1, 50)

VFuelRLift = np.zeros((len(NLiftEarliest), len(rLift)))
tLapRLift = np.zeros((len(NLiftEarliest), len(rLift)))

for i in range(0, len(NLiftEarliest)):
    for k in range(1, len(rLift)):
        tLapRLift[i, k], VFuelRLift[i, k], R = costFcn([rLift[k]], copy.deepcopy(c), [NLiftEarliest[i]], [NBrake[i]], None, False)

# get fuel consumption and lap time differences compared to original lap
tLapRLift = tLapRLift - c['tLap'][-1]
VFuelRLift = VFuelRLift - c['VFuel'][-1]

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
tLapMaxSave, VFuelMaxSave, R = costFcn(np.ones(len(NLiftEarliest)), c, NLiftEarliest, NBrake, None, False)

# optimisation for 100 steps between maximum lift and push
VFuelTGT = np.max([3.1, VFuelMaxSave])
VFuelTGT = np.linspace(VFuelMaxSave, c['VFuel'][-1], 100)

print('\nMaximum Lift:')
print('LapTime :', np.round(tLapMaxSave, 3))
print('VFuel :', np.round(VFuelMaxSave, 3))

# bounds and constaints
bounds = [(0, 1)]*6
LiftPointsVsFuelCons = {'VFuelTGT': np.empty((len(VFuelTGT), 1)), 'LiftPoints': np.empty((len(VFuelTGT), len(NLiftEarliest)))}

result = []
fun = []

for i in range(0, len(VFuelTGT)):  # optimisation loop

    VFuelConsTGT = VFuelTGT[i] - c['VFuel'][-1]

    FuelConstraint = {'type': 'eq', 'fun': calcFuelConstraint, 'args': (VFuelPolyFit, VFuelConsTGT)}

    temp_result = scipy.optimize.minimize(objectiveLapTime, [0.0]*6, args=(tLapPolyFit, VFuelPolyFit), method='SLSQP', bounds=bounds, constraints=FuelConstraint, options={'maxiter': 10000, 'ftol': 1e-09, 'iprint': 1, 'disp': False})

    result.append(temp_result)
    fun.append(temp_result['fun'])

    LiftPointsVsFuelCons['LiftPoints'][i, :] = result[i]['x']


LiftPointsVsFuelCons['VFuelTGT'] = VFuelTGT

plt.figure()
plt.title('tLap vs VFuelTGT')
plt.xlabel('VFuelTGT [l]')
plt.ylabel('dtLap [s]')
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
LiftPointsVsFuelCons['LapDistPct'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))
for i in range(0, len(NBrake)):  # lift zones
    # flip because x data must be monotonically increasing
    x = np.flip(1 - (np.linspace(NLiftEarliest[i], NBrake[i], NBrake[i]-NLiftEarliest[i]+1) - NLiftEarliest[i]) / (NBrake[i]-NLiftEarliest[i]))
    y = np.flip(c['LapDistPct'][np.linspace(NLiftEarliest[i], NBrake[i], NBrake[i]-NLiftEarliest[i]+1, dtype='int32')])
    for k in range(0, len(LiftPointsVsFuelCons['VFuelTGT'])):  # TGT
        LiftPointsVsFuelCons['LapDistPct'][k, i] = np.interp(LiftPointsVsFuelCons['LiftPoints'][k, i], x, y)

# transform back to original lap
temp = copy.deepcopy(c)
r = {}
keys = list(temp.keys())

for i in range(0, len(temp)):
    if keys[i] == 'tLap':
        r[keys[i]] = temp[keys[i]][NCut:-1] - d['tLap'][NApex[0]]
        r[keys[i]] = np.append(r[keys[i]], temp[keys[i]][0:NCut] + r[keys[i]][-1] + temp['dt'][-1])
    elif keys[i] == 'sLap':
        r[keys[i]] = temp[keys[i]][NCut:-1] - d['sLap'][NApex[0]]
        r[keys[i]] = np.append(r[keys[i]], temp[keys[i]][0:NCut] + r[keys[i]][-1] + temp['ds'][-1])
    elif keys[i] == 'LapDistPct':
        r[keys[i]] = temp[keys[i]][NCut:-1] - d['LapDistPct'][NApex[0]]
        r[keys[i]] = np.append(r[keys[i]], temp[keys[i]][0:NCut] + r[keys[i]][-1] + temp['dLapDistPct'][-1])
    else:
        r[keys[i]] = temp[keys[i]][NCut:-1]
        r[keys[i]] = np.append(r[keys[i]], temp[keys[i]][0:NCut])

# re-do calculations for cut lap
r['dt'] = np.diff(r['tLap'])
r['ds'] = np.diff(r['sLap'])
r['dLapDistPct'] = np.diff(r['LapDistPct'])
r = calcFuel(r)
r = calcLapTime(r)

# NApex = NApex[0:-1] + len(d['vCar']) - NCut - 1
NApex = NApex + len(d['vCar']) - NCut - 1
NApex[NApex > len(d['vCar'])] = NApex[NApex > len(d['vCar'])] - len(d['vCar']) + 1
NApex = np.sort(NApex)

NBrake = NBrake + len(d['vCar']) - NCut - 1
NBrake[NBrake > len(d['vCar'])] = NBrake[NBrake > len(d['vCar'])] - len(d['vCar']) + 1
NBrake = np.sort(NBrake)

NLiftEarliest = NLiftEarliest + len(d['vCar']) - NCut - 1
NLiftEarliest[NLiftEarliest > len(d['vCar'])] = NLiftEarliest[NLiftEarliest > len(d['vCar'])] - len(d['vCar']) + 1
NLiftEarliest = np.sort(NLiftEarliest)

NWOT = NWOT + len(d['vCar']) - NCut - 1
NWOT[NWOT > len(d['vCar'])] = NWOT[NWOT > len(d['vCar'])] - len(d['vCar']) + 1
NWOT = np.sort(NWOT)

plt.figure()
plt.plot(d['LapDistPct'], d['vCar'], label='original')
plt.plot(d['LapDistPct'], d['vCar'], label='new', linestyle='dashed')
plt.scatter(d['LapDistPct'][NApex], d['vCar'][NApex], label='NApex')
plt.scatter(d['LapDistPct'][NBrake], d['vCar'][NBrake], label='NBrake')
plt.scatter(d['LapDistPct'][NLiftEarliest], d['vCar'][NLiftEarliest], label='NLiftEarliest')
plt.scatter(d['LapDistPct'][NWOT], d['vCar'][NWOT], label='NWOT')
plt.legend()
plt.grid()
plt.show(block=False)

LiftPointsVsFuelCons['LapDistPct'] = LiftPointsVsFuelCons['LapDistPct'] + 100 - c['LapDistPct'][NCut]

# export data
saveJson(LiftPointsVsFuelCons)

print('Done')
