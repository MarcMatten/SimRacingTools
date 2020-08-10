import copy
import os
import time
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal

from functionalities.libs import filters, maths, importIBT, importExport
from libs.Car import Car

rhoFuel = 0.75  # TODO can this be obtained from IBT file?
nan = float('nan')


def costFcn(rLift, car, *args):
    C = args[0]  # data struct
    NLiftEarliest = args[1]  # index of ealiest lift points
    NBrake = args[2]  # index of braking points
    VFuelTgt = args[3]  # fuel target for optimistion mode
    optim = args[4]  # optimisation mode -> calc error?
    LiftGear = args[5]
    for i in range(0, len(NLiftEarliest)):
        C = stepFwds(C, NLiftEarliest[i] + int((1 - rLift[i]) * (NBrake[i] - NLiftEarliest[i])), LiftGear[i], car)

    C = calcFuel(C)
    C = calcLapTime(C)

    if optim:
        return C['tLap'][-1] + abs(VFuelTgt - C['VFuel'][-1]) * 1000
    else:
        return C['tLap'][-1], C['VFuel'][-1], C


def stepFwds(x, n, LiftGear, car):
    temp = copy.deepcopy(x)
    i = 0

    while temp['vCar'][n] < temp['vCar'][n + 1]:
        vCar = temp['vCar'][n]
        gLong = maths.polyVal(vCar, np.array(car.Coasting['gLongCoastPolyFit'][LiftGear])) - np.sin(temp['aTrackIncline'][n] / 180 * np.pi)
        ds = temp['ds'][n]
        temp['dt'][n] = -vCar / gLong - np.sqrt(np.square(vCar / gLong) + 2 * ds / gLong)
        temp['vCar'][n + 1] = temp['vCar'][n] + gLong * temp['dt'][n]
        temp['QFuel'][n] = maths.polyVal(vCar, np.array(car.Coasting['QFuelCoastPolyFit'][LiftGear]))
        n = n + 1
        i += 1

    return temp


def calcFuel(x):
    x['mFuel'] = np.cumsum(np.append(0, x['dt'] * x['QFuel'][0:-1]))/1000
    x['VFuel'] = x['mFuel']/rhoFuel
    return x


def calcLapTime(x):
    x['tLap'] = np.cumsum(np.append(0, x['dt']))
    return x


def stepBwds(x, i, LiftGear, car):

    n = i
    temp = copy.deepcopy(x)

    while temp['vCar'][n] <= x['vCar'][n-1]:
        vCar = temp['vCar'][n]
        gLong = maths.polyVal(vCar, np.array(car.Coasting['gLongCoastPolyFit'][LiftGear])) - np.sin(temp['aTrackIncline'][n] / 180 * np.pi)
        ds = temp['ds'][n]
        temp['dt'][n] = -vCar / gLong - np.sqrt(np.square(vCar / gLong) + 2 * ds / gLong)
        temp['vCar'][n - 1] = temp['vCar'][n] - gLong * temp['dt'][n]
        # temp['QFuel'][n] = 0.58
        temp['QFuel'][n] = maths.polyVal(vCar, np.array(car.Coasting['QFuelCoastPolyFit'][LiftGear]))
        n = n - 1

    return temp, int(n)


def objectiveLapTime(rLift, *args):
    tLapPolyFit = args[0]
    dt = 0
    for i in range(0, len(rLift)):
        dt = dt + maths.polyVal(rLift[i], tLapPolyFit[i, 0], tLapPolyFit[i, 1], tLapPolyFit[i, 2], tLapPolyFit[i, 3], tLapPolyFit[i, 4], tLapPolyFit[i, 5])
    return dt


def calcFuelConstraint(rLift, *args):
    VFuelPolyFit = args[0]
    VFuelConsTGT = args[1]
    dVFuel = 0
    for i in range(0, len(rLift)):
        dVFuel = dVFuel + maths.polyVal(rLift[i], VFuelPolyFit[i, 0], VFuelPolyFit[i, 1], VFuelPolyFit[i, 2], VFuelPolyFit[i, 3], VFuelPolyFit[i, 4], VFuelPolyFit[i, 5])
    return dVFuel - VFuelConsTGT


def saveJson(x, path):
    filepath = filedialog.asksaveasfilename(initialdir=path,
                                            initialfile='fuelSaving.json',
                                            title="Where to save results?",
                                            filetypes=[("JSON files", "*.json"), ("all files", "*.*")])

    if not filepath:
        print(time.strftime("%H:%M:%S", time.localtime()) + ':\tNo valid path provided...aborting!')
        return

    importExport.saveJson(x, filepath)

    print(time.strftime("%H:%M:%S", time.localtime()) + ':\tSaved data ' + filepath)


def createTrack(x):  # TODO: outsource to separate library, same code as in iDDUcalc?
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


def optimise(dirPath):
    BPlot = True

    root = tk.Tk()
    root.withdraw()
    
    # get ibt path
    ibtPath = filedialog.askopenfilename(initialdir=dirPath, title="Select IBT file",
                                         filetypes=(("IBT files", "*.ibt"), ("all files", "*.*")))

    if not ibtPath:
        print(time.strftime("%H:%M:%S", time.localtime()) + ':\tNo valid path to ibt file provided...aborting!')
        return

    # imoport ibt file
    d, _ = importIBT.importIBT(ibtPath,
                               lap='f',
                               channels=['zTrack', 'LapDistPct', 'rThrottle', 'rBrake', 'QFuel', 'SessionTime', 'VelocityX', 'VelocityY' ,'Yaw', 'Gear'],
                               channelMapPath=dirPath+'/functionalities/libs/iRacingChannelMap.csv')  # TODO: check if data is sufficient

    DriverCarIdx = d['DriverInfo']['DriverCarIdx']
    carScreenNameShort = d['DriverInfo']['Drivers'][DriverCarIdx]['CarScreenNameShort']
    TrackDisplayShortName = d['WeekendInfo']['TrackDisplayShortName']

    # If car file exists, load it. Otherwise, throw an error. TODO: whole section is duplicate with rollOut
    car = Car(carScreenNameShort)
    carFilePath = dirPath + '/data/car/' + carScreenNameShort + '.json'

    if carScreenNameShort + '.json' in importExport.getFiles(dirPath + '/data/car', 'json'):
        car.load(carFilePath)
    else:
        print('Error! Car file for {} does not exist. Create car with roll-out curves first!'.format(carScreenNameShort))

    # TODO: check if car has roll-out curve

    # create results directory
    resultsDirPath = dirPath + "/data/fuelSaving/" + TrackDisplayShortName + ' - ' + carScreenNameShort
    if not os.path.exists(resultsDirPath):
        os.mkdir(resultsDirPath)

    # calculate aTrackIncline smooth(atan( derivative('Alt' [m]) / derivative('Lap Distance' [m]) ), 1.5)
    d['aTrackIncline'] = np.arctan(np.gradient(d['zTrack']) / np.gradient(d['sLap']))
    d['aTrackIncline2'] = np.arctan(np.gradient(filters.movingAverage(d['zTrack'], 25)) / np.gradient(d['sLap']))

    d['sLap'] = d['sLap'] - d['sLap'][0]  # correct distance data

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

    plt.ioff()
    plt.figure()  # TODO: make plot nice
    plt.plot(d['sLap'], d['vCar'], 'k', label='Speed')
    plt.plot(d['sLap'], d['rThrottle'] * 10, 'g', label='rThrottle')
    plt.plot(d['sLap'], d['rBrake'] * 10, 'r', label='rBrake')
    plt.grid()
    plt.title('Apex Points')
    plt.xlabel('sLap [m]')
    plt.ylabel('vCar [m/s]')
    plt.scatter(d['sLap'][NApex], d['vCar'][NApex], label='Apex Points')
    plt.legend()
    plt.savefig(resultsDirPath + '/apexPoints.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

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
            if type(temp[keys[i]]) is dict:
                c[keys[i]] = temp[keys[i]]
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
    c['rBrake'][0] = 1  # fudging around to find the first brake point

    # find potential lift point (from full throttle to braking)
    NWOT = scipy.signal.find_peaks(c['rThrottle'], height=1, plateau_size=20)
    NBrake = scipy.signal.find_peaks(1 - np.clip(c['rBrake'], 0.1, 1), plateau_size=(30, 10000), height=0.9)

    # sections for potential lifting
    NWOT = NWOT[1]['left_edges']
    NBrake = NBrake[1]['right_edges']

    # elimination obsolete points
    NApexNew = []
    NBrakeNew = []
    NWOTNew = []

    for i in range(0, len(NApex)):
        k = len(NApex) - 1 - i
        if NApex[k] > np.min(NBrake):
            NApexNew.append(NApex[k])

            for j in range(0, len(NBrake)):
                l = len(NBrake) - 1 - j
                if NBrake[l] < NApex[k]:
                    NBrakeNew.append(NBrake[l])

                    for m in range(0, len(NWOT)):
                        n = len(NWOT) - 1 - m
                        if NWOT[n] < NBrake[l]:
                            NWOTNew.append(NWOT[n])

                            break

                    break

    del i, k, m, n, l, j

    NApex = np.flip(NApexNew)
    NBrake = np.flip(NBrakeNew)
    NWOT = np.flip(NWOTNew)

    LiftGear = c['Gear'][NBrake]

    plt.figure()  # TODO: make plot nice
    plt.plot(c['sLap'], c['vCar'], label='Speed - Push Lap')
    plt.scatter(c['sLap'][NWOT], c['vCar'][NWOT], label='Full Throttle Points')
    plt.scatter(c['sLap'][NBrake], c['vCar'][NBrake], label='Brake Points')
    plt.scatter(c['sLap'][NApex], c['vCar'][NApex], label='Apex Points')

    plt.grid()
    plt.legend()
    plt.title('Sections')
    plt.xlabel('sLap [m]')
    plt.ylabel('vCar [m/s]')
    plt.savefig(resultsDirPath + '/sections.png', dpi=300, orientation='landscape', progressive=True)

    print('\nPush Lap:')
    print('LapTime :', np.round(c['tLap'][-1], 3))
    print('VFuel :', np.round(c['VFuel'][-1], 3))

    # Find earliest lift points. Assumption: arriving at apex with apex speed but no brake application
    NLiftEarliest = np.array([], dtype='int32')
    c_temp = copy.deepcopy(c)
    for i in range(0, len(NWOT)):
        c_temp, n = stepBwds(c_temp, NBrake[i] + int(0.85*(NApex[i]-NBrake[i])), LiftGear[i], car)
        NLiftEarliest = np.append(NLiftEarliest, n)

    plt.figure()  # TODO: make plot nice
    plt.title('Earliest Lift Points')
    plt.xlabel('sLap [m]')
    plt.ylabel('vCar [m/s]')
    plt.plot(c['sLap'], c['vCar'], label='Speed')
    plt.plot(c_temp['sLap'], c_temp['vCar'], label='Maximum Lifting')
    plt.scatter(c_temp['sLap'][NLiftEarliest], c_temp['vCar'][NLiftEarliest], label='Earliest Lift Point')
    plt.grid()
    plt.legend()
    plt.savefig(resultsDirPath + '/earliestLift.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    # for each lifting zone calculate various ratios of lifting
    rLift = np.linspace(0, 1, 50)

    VFuelRLift = np.zeros((len(NLiftEarliest), len(rLift)))
    tLapRLift = np.zeros((len(NLiftEarliest), len(rLift)))

    for i in range(0, len(NLiftEarliest)):
        for k in range(1, len(rLift)):
            tLapRLift[i, k], VFuelRLift[i, k], R = costFcn([rLift[k]], car, copy.deepcopy(c), [NLiftEarliest[i]], [NBrake[i]], None, False, [LiftGear[i]])

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

        tLapPolyFit[i, :], temp = scipy.optimize.curve_fit(maths.polyVal, x, t, [0]*6)
        VFuelPolyFit[i, :], temp = scipy.optimize.curve_fit(maths.polyVal, x, f, [0]*6)

        if BPlot:  # TODO: save these plots in a subdirectory
            plt.figure()  # TODO: make plot nice
            plt.title('Lap Time Loss - Lift Zone ' + str(i + 1))
            plt.xlabel('rLift [-]')
            plt.ylabel('dtLap [s]')
            plt.scatter(rLift, tLapRLift[i, :])
            plt.plot(rLiftPlot, maths.polyVal(rLiftPlot, tLapPolyFit[i, 0], tLapPolyFit[i, 1], tLapPolyFit[i, 2], tLapPolyFit[i, 3], tLapPolyFit[i, 4], tLapPolyFit[i, 5]))
            plt.grid()
            plt.savefig(resultsDirPath + '/timeLoss_LiftZone_' + str(i + 1) + '.png', dpi=300, orientation='landscape', progressive=True)
            plt.close()

            plt.figure()  # TODO: make plot nice
            plt.title('Fuel Save - Lift Zone ' + str(i + 1))
            plt.xlabel('rLift [-]')
            plt.ylabel('dVFuel [l]')
            plt.scatter(rLift, VFuelRLift[i, :])
            plt.plot(rLiftPlot, maths.polyVal(rLiftPlot, VFuelPolyFit[i, 0], VFuelPolyFit[i, 1], VFuelPolyFit[i, 2], VFuelPolyFit[i, 3], VFuelPolyFit[i, 4], VFuelPolyFit[i, 5]))
            plt.grid()
            plt.savefig(resultsDirPath + '/fuelSave_LiftZone_' + str(i + 1) + '.png', dpi=300, orientation='landscape', progressive=True)
            plt.close()

    # maximum lift
    tLapMaxSave, VFuelMaxSave, R = costFcn(np.ones(len(NLiftEarliest)), car, c, NLiftEarliest, NBrake, None, False, LiftGear)

    # optimisation for 100 steps between maximum lift and push
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

        # actual optimisation
        temp_result = scipy.optimize.minimize(objectiveLapTime, np.zeros(6), args=(tLapPolyFit, VFuelPolyFit), method='SLSQP', bounds=bounds, constraints=FuelConstraint,
                                              options={'maxiter': 10000, 'ftol': 1e-09, 'iprint': 1, 'disp': False})

        result.append(temp_result)
        fun.append(temp_result['fun'] + 0.029)

        LiftPointsVsFuelCons['LiftPoints'][i, :] = result[i]['x']


    LiftPointsVsFuelCons['VFuelTGT'] = VFuelTGT

    plt.figure()  # TODO: make plot nice
    plt.title('Detla tLap vs VFuel')
    plt.xlabel('VFuel [l]')
    plt.ylabel('Delta tLap [s]')
    plt.plot(LiftPointsVsFuelCons['VFuelTGT'], fun, label='Simulation')
    plt.grid()
    plt.legend()
    plt.savefig(resultsDirPath + '/DetlatLap_vs_VFuel.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    plt.figure()  # TODO: make plot nice
    plt.title('rLift vs VFuelTGT')
    plt.xlabel('VFuelTGT [l]')
    plt.ylabel('rLift [-]')
    for k in range(0, len(NLiftEarliest)):
        plt.plot(LiftPointsVsFuelCons['VFuelTGT'], LiftPointsVsFuelCons['LiftPoints'][:, k], label='Lift Zone ' + str(k+1))

    plt.legend()
    plt.grid()
    plt.savefig(resultsDirPath + '/rLift_vs_vFuelTGT.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    # get LapDistPct
    LiftPointsVsFuelCons['LapDistPct'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))
    for i in range(0, len(NBrake)):  # lift zones
        # flip because x data must be monotonically increasing
        x = np.flip(1 - (np.linspace(NLiftEarliest[i], NBrake[i], NBrake[i]-NLiftEarliest[i]+1) - NLiftEarliest[i]) / (NBrake[i]-NLiftEarliest[i]))
        y = np.flip(c['LapDistPct'][np.linspace(NLiftEarliest[i], NBrake[i], NBrake[i]-NLiftEarliest[i]+1, dtype='int32')])
        for k in range(0, len(LiftPointsVsFuelCons['VFuelTGT'])):  # TGT
            LiftPointsVsFuelCons['LapDistPct'][k, i] = np.interp(LiftPointsVsFuelCons['LiftPoints'][k, i], x, y)

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

    plt.figure()  # TODO: make plot nice
    plt.plot(d['LapDistPct'], d['vCar'], label='original')
    plt.plot(d['LapDistPct'], d['vCar'], label='new', linestyle='dashed')
    plt.scatter(d['LapDistPct'][NApex], d['vCar'][NApex], label='NApex')
    plt.scatter(d['LapDistPct'][NBrake], d['vCar'][NBrake], label='NBrake')
    plt.scatter(d['LapDistPct'][NLiftEarliest], d['vCar'][NLiftEarliest], label='NLiftEarliest')
    plt.scatter(d['LapDistPct'][NWOT], d['vCar'][NWOT], label='NWOT')
    plt.legend()
    plt.grid()
    plt.savefig(resultsDirPath + '/resultsCheck.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    plt.figure()  # TODO: make plot nice
    plt.plot(d['x'], d['y'], label='Track')
    plt.scatter(d['x'][NApex], d['y'][NApex], label='NApex')
    plt.scatter(d['x'][NBrake], d['y'][NBrake], label='NBrake')
    plt.scatter(d['x'][NLiftEarliest], d['y'][NLiftEarliest], label='NLiftEarliest')
    plt.scatter(d['x'][NWOT], d['y'][NWOT], label='NWOT')
    plt.legend()
    plt.grid()
    plt.savefig(resultsDirPath + '/trackMap.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    LiftPointsVsFuelCons['LapDistPct'] = LiftPointsVsFuelCons['LapDistPct'] + 100 - c['LapDistPct'][NCut]
    LiftPointsVsFuelCons['SetupName'] = d['DriverInfo']['DriverSetupName']
    LiftPointsVsFuelCons['CarSetup'] = d['CarSetup']

    # export data
    saveJson(LiftPointsVsFuelCons, resultsDirPath)

    print(time.strftime("%H:%M:%S", time.localtime()) + ':\tCompleted Fuel Saving Optimisation!')
