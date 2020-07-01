import csv
import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt
import copy
from costFcn import costFcn
import json
import irsdk
from importIBT import importIBT
import tkinter as tk
from tkinter import filedialog

rhoFuel = 0.75

def calcFuel(x):
    x['mFuel'] = np.cumsum(np.append(0, x['dt'] * x['QFuel'][0:-1]))/1000
    x['VFuel'] = x['mFuel']/rhoFuel
    return x


def calcLapTime(x):
    x['tLap'] = np.cumsum(np.append(0, x['dt']))
    return x


def stepBwds(x, i, LiftGear):

    n = i
    temp = copy.deepcopy(x)

    while temp['vCar'][n] <= x['vCar'][n-1]:
        vCar = temp['vCar'][n]
        gLong = getGLong(vCar, temp['aTrackIncline'][n], LiftGear)
        ds = temp['ds'][n]
        temp['dt'][n] = -vCar/gLong - np.sqrt( np.square(vCar / gLong) + 2*ds/gLong )
        temp['vCar'][n-1] = temp['vCar'][n] - gLong * temp['dt'][n]
        temp['QFuel'][n] = 0.58
        n = n-1

    return temp, int(n)


def getGLong(v, aTrackIncline, LiftGear):
    if LiftGear == 6:
        return -0.0010998*np.square(v) + 0.057576 * v - 2.0634 - np.sin(aTrackIncline/180*np.pi)
    else:  # 4
        return -0.0007912*np.square(v) + 0.0008683 * v - 0.41181 - np.sin(aTrackIncline/180*np.pi)


def poly6(x, a, b, c, d, e, f):
    return a + b * x + c * np.power(x, 2) + d * np.power(x, 3) + e * np.power(x, 4) + f * np.power(x, 5)


def objectiveLapTime(rLift, *args):
    tLapPolyFit = args[0]
    dt = 0
    for i in range(0, len(rLift)):
        dt = dt + poly6(rLift[i], tLapPolyFit[i, 0], tLapPolyFit[i, 1], tLapPolyFit[i, 2], tLapPolyFit[i, 3], tLapPolyFit[i, 4], tLapPolyFit[i, 5])
    return dt


def calcFuelConstraint(rLift, *args):
    VFuelPolyFit = args[0]
    VFuelConsTGT = args[1]
    dVFuel = 0
    for i in range(0, len(rLift)):
        dVFuel = dVFuel + poly6(rLift[i], VFuelPolyFit[i, 0], VFuelPolyFit[i, 1], VFuelPolyFit[i, 2], VFuelPolyFit[i, 3], VFuelPolyFit[i, 4], VFuelPolyFit[i, 5])
    return dVFuel - VFuelConsTGT


def saveJson(x):
    filepath = 'FuelTGTLiftPoints.json'

    variables = list(x.keys())

    data = {}

    for i in range(0, len(variables)):
        data[variables[i]] = x[variables[i]].transpose().tolist()

    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

    print('Saved data ' + filepath)

# def moving_average(a, n) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


def moving_average(a, n):
    temp = a[-n:]
    temp = np.append(temp, a)
    temp = np.append(temp, a[0:n])
    r = np.zeros(np.size(temp))
    for i in range(0, len(temp)):
        if i < n:
            r[i] = np.mean(temp[0:i+n])
        elif len(temp) < i + n:
            r[i] = np.mean(temp[i-n:])
        else:
            r[i] = np.mean(temp[i-n:n+i])
    return r[n:-n]


def createTrack(x):
    dx = np.array(0)
    dy = np.array(0)

    dist = x['LapDistPct'] * 100

    dist[0] = 0
    dist[-1] = 100

    dx = np.append(dx, np.cos(x['Yaw'][0:-1]) * x['VelocityX'][0:-1] * x['dt'] - np.sin(x['Yaw'][0:-1]) * x['VelocityY'][0:-1] * x['dt'])
    dy = np.append(dy, np.cos(x['Yaw'][0:-1]) * x['VelocityY'][0:-1] * x['dt'] + np.sin(x['Yaw'][0:-1]) * x['VelocityX'][0:-1] * x['dt'])

    tempx = np.cumsum(dx, dtype=float).tolist()
    tempy = np.cumsum(dy, dtype=float).tolist()

    xError = tempx[-1] - tempx[0]
    yError = tempy[-1] - tempy[0]

    tempdx = np.array(0)
    tempdy = np.array(0)

    tempdx = np.append(tempdx, dx[1:len(dx)] - xError / (len(dx) - 1))
    tempdy = np.append(tempdy, dy[1:len(dy)] - yError / (len(dy) - 1))

    x = np.cumsum(tempdx, dtype=float)
    y = np.cumsum(tempdy, dtype=float)

    x[-1] = 0
    y[-1] = 0

    return x, y


VFueliRacing = np.array([3.255, 3.264, 3.271, 3.218, 3.224, 3.182, 3.106, 3.098, 3.005, 3.026, 3.003, 2.904, 2.927, 2.826, 2.817, 2.759])
tLapiRacing = np.array([105.488, 105.272, 105.471, 105.461, 105.454, 105.600, 105.608, 105.713, 105.820, 105.636, 105.635, 106.029, 105.939, 106.393, 106.309, 106.463])-105.35

BPlot = True
nan = float('nan')

# # import CSV files with the following data: "Time", "Ground Speed", "G Force Lat", "G Force Long", "ThrottleRaw", "BrakeRaw", "FuelUsePerHour", "Fuel Density", "Lap Distance", "FuelLevel",
# # "LapDistPct", "aTrackIncline", "GPS Altitude"
# # filename = 'fordgt2017_monza full 2020-04-04 15-09-18_Stint_1.csv'
# filename = 'fordgt2017_monza full 2020-05-14 20-55-54_Stint_1.csv'
# header = ["tLap", "vCar", "gLat", "gLong", "rThrottle", "rBrake", "QFuel", "rhoFuel", "sLap", "FuelLevel", "LapDistPct", "aTrackIncline", "GPSAltitude", "Gear"]
#
# # import ibt file
#
# # declare baseline data dict and fill it from CSV
# d2 = {}
#
# for i in range(0, len(header)):
#     d2[header[i]] = np.array([])
#
# with open(filename) as csv_file:
#     csv_reader = csv.reader(csv_file)
#     for line in csv_reader:
#         for i in range(0, len(header)):
#             d2[header[i]] = np.append(d2[header[i]], float(line[i]))

# import ibt file
# MyIbtPath = 'C:/Users/Marc/Documents/Projekte/SimRacingTools/fordgt2017_monza full 2020-05-14 20-55-54.ibt'
root = tk.Tk()
root.withdraw()
MyIbtPath = filedialog.askopenfilename(initialdir="C:/Users/Marc/Documents/Projekte/SimRacingTools/FuelSavingOptimiser", title="Select IBT file", filetypes=(("IBT files", "*.ibt"), ("all files", "*.*")))

# MyChannelMap = {'Speed': ['vCar', 1],               # m/s
#               'LapCurrentLapTime': ['tLap', 1],     # s
#               'LatAccel': ['gLat', 1],              # m/s²
#               'LongAccel': ['gLong', 1],            # m/s²
#               'ThrottleRaw': ['rThrottle', 1],      # 1
#               'BrakeRaw': ['rBrake', 1],            # 1
#               'FuelUsePerHour': ['QFuel', 1/3.6],   # l/h --> g/s
#               'LapDist': ['sLap', 1],               # m
#               'Alt': ['GPSAltitude', 1]             # m,
#               }

d = importIBT(MyIbtPath, 'f')

# calculate aTrackIncline smooth(atan( derivative('Alt' [m]) / derivative('Lap Distance' [m]) ), 1.5)
d['aTrackIncline'] = np.arctan(np.gradient(d['GPSAltitude'])/np.gradient(d['sLap']))
d['aTrackIncline2'] = np.arctan(np.gradient(moving_average(d['GPSAltitude'], 25))/np.gradient(d['sLap']))

d['sLap'] = d['sLap'] - d['sLap'][0]  # correct distance data
# d['LapDistPct'] = d['LapDistPct'] - d['LapDistPct'][0]
# d['LapDistPct'] = d['LapDistPct'] / d['LapDistPct'][-1] *100

# do some calculations for baseline data
d['dt'] = np.diff(d['SessionTime'])
d['ds'] = np.diff(d['sLap'])
d['dLapDistPct'] = np.diff(d['LapDistPct'])
d = calcFuel(d)
d = calcLapTime(d)

d['x'], d['y'] = createTrack(d)

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
NCut = len(temp[keys[0]][NApex[0]:-1])

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
NApex = np.append(NApex, len(c['tLap'])-1)  # fake apex for last index
c['rBrake'][0] = 0.001  # fudging around to find the first brake point

# find potential lift point (from full throttle to braking)
NWOT = scipy.signal.find_peaks(c['rThrottle'], height=1, plateau_size=80)
NBrake = scipy.signal.find_peaks(1-c['rBrake'], height=1, plateau_size=40)

if not len(NWOT[1]) == len(NBrake[1]):
    print('Error! Number of brake application and full throttle point don nit match!')
    quit()

# sections for potential lifting
NWOT = NWOT[1]['left_edges']
NBrake = NBrake[1]['right_edges']
LiftGear = c['Gear'][NBrake]

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
    c_temp, n = stepBwds(c_temp, NBrake[i] + int(0.85*(NApex[i]-NBrake[i])), LiftGear[i])
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
        tLapRLift[i, k], VFuelRLift[i, k], R = costFcn([rLift[k]], copy.deepcopy(c), [NLiftEarliest[i]], [NBrake[i]], None, False, [LiftGear[i]])

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
tLapMaxSave, VFuelMaxSave, R = costFcn(np.ones(len(NLiftEarliest)), c, NLiftEarliest, NBrake, None, False, LiftGear)

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
    fun.append(temp_result['fun']+0.029)

    LiftPointsVsFuelCons['LiftPoints'][i, :] = result[i]['x']


LiftPointsVsFuelCons['VFuelTGT'] = VFuelTGT

plt.figure()
plt.title('Detla tLap vs VFuel')
plt.xlabel('VFuel [l]')
plt.ylabel('Delta tLap [s]')
plt.plot(LiftPointsVsFuelCons['VFuelTGT'], fun, label='Simulation')
plt.scatter(VFueliRacing, tLapiRacing, label='iRacing')
plt.grid()
plt.legend()
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

# correlation -------------------------
temp1, temp2, C = costFcn(LiftPointsVsFuelCons['LiftPoints'][68, :], copy.deepcopy(c), NLiftEarliest, NBrake, None, False, LiftGear)

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

# CORREELATION transform back to original lap
temp2 = copy.deepcopy(C)
R = {}
keys = list(temp2.keys())

for i in range(0, len(temp2)):
    if keys[i] == 'tLap':
        R[keys[i]] = temp2[keys[i]][NCut:-1] - d['tLap'][NApex[0]]
        R[keys[i]] = np.append(R[keys[i]], temp2[keys[i]][0:NCut] + R[keys[i]][-1] + temp2['dt'][-1])
    elif keys[i] == 'sLap':
        R[keys[i]] = temp2[keys[i]][NCut:-1] - d['sLap'][NApex[0]]
        R[keys[i]] = np.append(R[keys[i]], temp2[keys[i]][0:NCut] + R[keys[i]][-1] + temp2['ds'][-1])
    elif keys[i] == 'LapDistPct':
        R[keys[i]] = temp2[keys[i]][NCut:-1] - d['LapDistPct'][NApex[0]]
        R[keys[i]] = np.append(R[keys[i]], temp2[keys[i]][0:NCut] + R[keys[i]][-1] + temp2['dLapDistPct'][-1])
    else:
        R[keys[i]] = temp2[keys[i]][NCut:-1]
        R[keys[i]] = np.append(R[keys[i]], temp2[keys[i]][0:NCut])

# re-do calculations for cut lap
R['sLap'] = R['sLap'] - R['sLap'][0]
R['tLap'] = R['tLap'] - R['tLap'][0]
R['LapDistPct'] = R['LapDistPct'] - R['LapDistPct'][0]
R['dt'] = np.diff(R['tLap'])
R['ds'] = np.diff(R['sLap'])
R['dLapDistPct'] = np.diff(R['LapDistPct'])
R = calcFuel(R)
R = calcLapTime(R)


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



plt.figure()
plt.plot(d['x'], d['y'], label='Track')
plt.scatter(d['x'][NApex], d['y'][NApex], label='NApex')
plt.scatter(d['x'][NBrake], d['y'][NBrake], label='NBrake')
plt.scatter(d['x'][NLiftEarliest], d['y'][NLiftEarliest], label='NLiftEarliest')
plt.scatter(d['x'][NWOT], d['y'][NWOT], label='NWOT')
plt.legend()
plt.grid()
plt.show(block=False)

LiftPointsVsFuelCons['LapDistPct'] = LiftPointsVsFuelCons['LapDistPct'] + 100 - c['LapDistPct'][NCut]

# export data
saveJson(LiftPointsVsFuelCons)

# correlation -------------------------
# ibtPath = 'C:/Users/Marc/Documents/iRacing/Telemetry/fordgt2017_monza full 2020-05-14 21-20-28.ibt'
# ibtPath = 'C:/Users/Marc/Documents/Projekte/SimRacingTools/FuelSavingOptimiser/fordgt2017_monza full 2020-05-03 20-52-05.ibt'
#
#
# ir_ibt = irsdk.IBT()
# ir_ibt.open(ibtPath)
# i = dict()
# i['vCar'] = np.array(ir_ibt.get_all('Speed'))
# i['LapDistPct'] = np.array(ir_ibt.get_all('LapDistPct'))
# i['NLap'] = np.array(ir_ibt.get_all('Lap'))
# k = [i for i, x in enumerate(i['NLap']) if x == 34]
#
# plt.figure()
# plt.plot(R['LapDistPct']*100, R['vCar'], label='Simulation')
# plt.plot(i['LapDistPct'][k]*100, i['vCar'][k], label='iRacing')
# plt.legend()
# plt.grid()
# plt.show(block=False)

print('Done')
