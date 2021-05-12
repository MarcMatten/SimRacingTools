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

nan = float('nan')


def costFcn(rLift, car, *args):
    C = copy.deepcopy(args[0])  # data struct
    NLiftEarliest = args[1]  # index of ealiest lift points
    NBrake = args[2]  # index of braking points
    VFuelTgt = args[3]  # fuel target for optimistion mode
    optim = args[4]  # optimisation mode -> calc error?
    LiftGear = args[5]
    if len(args) >= 7:
        NReference = args[6]
    C['NLift'] = np.array([], dtype=int)
    for i in range(0, len(NLiftEarliest)):
        NLift = NLiftEarliest[i] + int((1 - rLift[i]) * (NBrake[i] - NLiftEarliest[i]))
        C['NLift'] = np.append(C['NLift'], NLift)
        if NBrake[i] == NLift:
            continue
        C = stepFwds(C, C['NLift'][i], LiftGear[i], car, NBrake[i])

    C = calcFuel(C)
    C = calcLapTime(C)

    if optim:
        return C['tLap'][-1] + abs(VFuelTgt - C['VFuel'][-1]) * 1000
    else:
        if len(args) >= 7:
            C['VFuelLift'] = C['VFuel'][C['NLift']]
            C['VFuelReference'] = C['VFuel'][NReference]
        return C['tLap'][-1], C['VFuel'][-1], C


def stepFwds(x, n, LiftGear, car, NBrake):
    temp = copy.deepcopy(x)
    i = 0

    while n < NBrake or temp['vCar'][n] < temp['vCar'][n + 1] and n + 1 < len(temp['vCar']):
        vCar = temp['vCar'][n]
        gLong = maths.polyVal(vCar, np.array(car.Coasting['gLongCoastPolyFit'][LiftGear])) - np.sin(temp['aTrackIncline'][n] / 180 * np.pi)
        ds = temp['ds'][n]
        temp['dt'][n] = -vCar / gLong - np.sqrt(np.square(vCar / gLong) + 2 * ds / gLong)
        temp['vCar'][n + 1] = temp['vCar'][n] + gLong * temp['dt'][n]
        temp['QFuel'][n] = maths.polyVal(vCar, np.array(car.Coasting['QFuelCoastPolyFit'][LiftGear]))
        n += 1
        i += 1

        if n == len(temp['vCar']) - 1:
            break

    return temp


def calcFuel(x):
    x['VFuel'] = np.cumsum(np.append(0, x['dt'] * x['QFuel'][0:-1]))
    x['mFuel'] = x['VFuel'] * x['DriverInfo']['DriverCarFuelKgPerLtr']
    return x


def calcLapTime(x):
    x['tLap'] = np.cumsum(np.append(0, x['dt']))
    return x


def stepBwds(x, i, LiftGear, car, NApex_temp):
    n = i
    temp = copy.deepcopy(x)

    if NApex_temp == len(temp['vCar']) - 1:
        NApex_temp = 0

    while temp['vCar'][n] <= x['vCar'][n - 1] and n > NApex_temp:
        vCar = temp['vCar'][n]
        gLong = maths.polyVal(vCar, np.array(car.Coasting['gLongCoastPolyFit'][LiftGear])) - np.sin(temp['aTrackIncline'][n] / 180 * np.pi)
        ds = temp['ds'][n]
        temp['dt'][n] = -vCar / gLong - np.sqrt(np.square(vCar / gLong) + 2 * ds / gLong)
        temp['vCar'][n - 1] = temp['vCar'][n] - gLong * temp['dt'][n]
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


def optimise(dirPath, TelemPath):
    BPlot = True

    root = tk.Tk()
    root.withdraw()

    # get ibt path
    ibtPath = filedialog.askopenfilename(initialdir=TelemPath, title="Select IBT file",
                                         filetypes=(("IBT files", "*.ibt"), ("all files", "*.*")))

    if not ibtPath:
        print(time.strftime("%H:%M:%S", time.localtime()) + ':\tNo valid path to ibt file provided...aborting!')
        return

    # imoport ibt file
    d, _ = importIBT.importIBT(ibtPath,
                               lap='f',
                               channels=['zTrack', 'LapDistPct', 'rThrottle', 'rBrake', 'QFuel', 'SessionTime', 'VelocityX', 'VelocityY', 'Yaw', 'Gear', 'VFuel'],
                               channelMapPath=dirPath + '/functionalities/libs/iRacingChannelMap.csv')  # TODO: check if data is sufficient

    DriverCarIdx = d['DriverInfo']['DriverCarIdx']
    carScreenNameShort = d['DriverInfo']['Drivers'][DriverCarIdx]['CarScreenNameShort']
    # TrackDisplayShortName = d['WeekendInfo']['TrackDisplayShortName']
    TrackName = d['WeekendInfo']['TrackName']

    # If car file exists, load it. Otherwise, throw an error. TODO: whole section is duplicate with rollOut
    car = Car(carScreenNameShort)
    carFilePath = dirPath + '/data/car/' + carScreenNameShort + '.json'

    if carScreenNameShort + '.json' in importExport.getFiles(dirPath + '/data/car', 'json'):
        car.load(carFilePath)
    else:
        print('Error! Car file for {} does not exist. Create car with roll-out curves first!'.format(carScreenNameShort))

    # TODO: check if car has roll-out curve

    # create results directory
    resultsDirPath = dirPath + "/data/fuelSaving/" + TrackName + ' - ' + carScreenNameShort
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

    d['x'], d['y'] = maths.createTrack(d)

    print('\nPreperations done ... Analysing track')

    # find apex points
    NApex = scipy.signal.find_peaks(200 - d['vCar'], height=10, prominence=1)
    NApex = NApex[0]
    dNApex = np.diff(d['sLap'][NApex]) < 50
    if any(dNApex):
        for i in range(0, len(dNApex)):
            if dNApex[i]:
                NApex = np.delete(NApex, i + 1)

    print('NApex: ', NApex)
    print('LapDistPctApex: ', d['LapDistPct'][NApex])

    print('\tFound {} apex points'.format(len(NApex)))
    # print('NApex: ',  NApex)
    # print('dNApex: ',  np.diff(NApex))

    plt.ioff()
    plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    plt.tight_layout()
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

    d['NApex'] = NApex

    # cut lap at first apex
    print('\tCutting data at {} m'.format(np.round(d['sLap'][NApex][0], 1)))
    # create new data dict for cut lap --> c
    temp = copy.deepcopy(d)
    c = {}
    keys = list(temp.keys())
    NCut = len(temp[keys[10]][NApex[0]:-1])

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
    NApex = NApex[1:] - NApex[0]
    NApex = np.append(NApex, len(c['tLap']) - 1)  # fake apex for last index
    c['rBrake'][0] = 1  # fudging around to find the first brake point

    # find potential lift point (from full throttle to braking)
    NWOT = scipy.signal.find_peaks(c['rThrottle'], height=1, plateau_size=20)
    # NBrake = scipy.signal.find_peaks(1 - np.clip(c['rBrake'], 0.1, 1), plateau_size=(30, 10000), height=0.8, prominence=0.25)
    NBrake = scipy.signal.find_peaks(np.clip(c['rBrake'], 0.01, 0.02)*100, plateau_size=(10, 1000))

    # sections for potential lifting
    NWOT = NWOT[1]['left_edges']
    # NBrake = NBrake[1]['right_edges']
    NBrake = NBrake[1]['left_edges']

    c['NApex'] = NApex

    # elimination obsolete points
    NApexNew = []
    NBrakeNew = []
    NWOTNew = []

    for i in range(0, len(NApex)):
        k = len(NApex) - 1 - i
        if NApex[k] > np.min(NBrake):
            NApexNew = np.append(NApexNew, NApex[k])
            BNBrakeNew = False
            BNWOTNew = False

            for j in range(0, len(NBrake)):
                l = len(NBrake) - 1 - j
                if NBrake[l] < NApexNew[-1]:
                    if not BNBrakeNew:
                        if k > 0:
                            if not NBrake[l] < NApex[k - 1]:
                                NBrakeNew = np.append(NBrakeNew, NBrake[l])
                                BNBrakeNew = True
                            else:
                                NApexNew = np.delete(NApexNew, len(NApexNew) - 1)
                                break
                        else:
                            NBrakeNew = np.append(NBrakeNew, NBrake[l])
                            BNBrakeNew = True

                    for m in range(0, len(NWOT)):
                        n = len(NWOT) - 1 - m
                        if NWOT[n] < NApexNew[-1]:
                            if not BNWOTNew:
                                if n > 0 and k > 0:
                                    if not NWOT[n] < NApex[k - 1]:
                                        NWOTNew = np.append(NWOTNew, NWOT[n])
                                        BNWOTNew = True
                                        break
                                    else:
                                        NApexNew = np.delete(NApexNew, len(NApexNew) - 1)
                                        break
                                else:
                                    NWOTNew = np.append(NWOTNew, NWOT[n])
                                    BNWOTNew = True
                                    break
                        # else:
                        #     NWOTNew = np.append(NWOTNew, NWOT[n])
                        #     break

    # plt.scatter(c['sLap'][NApex], c['vCar'][NApex], label='Apex Points')
    del i, k, m, n, l, j

    print('\tIdentified reference points')

    NApex = np.flip(NApexNew).astype(int)
    NBrake = np.flip(NBrakeNew).astype(int)
    NReference = np.array(NBrake + (NApex - NBrake) * 0.5, dtype=int)
    NWOT = np.flip(NWOTNew).astype(int)

    c['NApex'] = NApex
    c['NBrake'] = NBrake
    c['NReference'] = NReference
    c['NWOT'] = NWOT

    LiftGear = c['Gear'][NBrake]

    plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    plt.tight_layout()
    plt.plot(c['sLap'], c['vCar'], label='Speed - Push Lap')
    plt.scatter(c['sLap'][NWOT], c['vCar'][NWOT], label='Full Throttle Points')
    plt.scatter(c['sLap'][NBrake], c['vCar'][NBrake], label='Brake Points')
    plt.scatter(c['sLap'][NApex], c['vCar'][NApex], label='Apex Points')
    plt.scatter(c['sLap'][NReference], c['vCar'][NReference], label='Reference Points')

    plt.grid()
    plt.legend()
    plt.title('Sections')
    plt.xlabel('sLap [m]')
    plt.ylabel('vCar [m/s]')
    plt.savefig(resultsDirPath + '/sections.png', dpi=300, orientation='landscape', progressive=True)

    # print('NApex: ', NApex)
    # print('sApex: ', c['sLap'][NApex])
    # print('NWOT: ', NWOT)
    # print('sWOT: ', c['sLap'][NWOT])
    # print('NBrake: ', NBrake)
    # print('sBrake: ', c['sLap'][NBrake])

    print('\nOptimisation Boundaries')
    print('\n\tPush Lap')

    # TODO: cald these reference values
    print('\tLapTime:\t{} s'.format(np.round(c['tLap'][-1], 3)))
    print('\tVFuel:\t\t{} l'.format(np.round(c['VFuel'][-1], 3)))

    # Find earliest lift points. Assumption: arriving at apex with apex speed but no brake application
    print('\n\tCalculating Lifting')
    NLiftEarliest = np.array([], dtype='int32')
    c_temp = copy.deepcopy(c)
    for i in range(0, len(NWOT)):
        if i == 0:
            NApex_temp = NApex[-1]
        else:
            NApex_temp = NApex[i - 1]

        c_temp, n = stepBwds(c_temp, NBrake[i] + int(0.85 * (NApex[i] - NBrake[i])), LiftGear[i], car, NApex_temp)
        NLiftEarliest = np.append(NLiftEarliest, n)

    NLiftEarliest = np.maximum(NWOT, NLiftEarliest)

    plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    plt.tight_layout()
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
        print('\t\tLift zone: {}/{}'.format(i + 1, len(NLiftEarliest)))
        for k in range(1, len(rLift)):
            # print('Lift zone: {} - rLift: {}'.format(i, k))
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

    NZone = np.roll(np.linspace(1, len(NLiftEarliest), len(NLiftEarliest),dtype=int),-1)

    for i in range(0, len(NLiftEarliest)):
        # remove nan indices
        yTime = tLapRLift[i, :]
        yFuel = VFuelRLift[i, :]
        x = rLift[~np.isnan(yTime)]
        f = yFuel[~np.isnan(yTime)]
        t = yTime[~np.isnan(yTime)]

        tLapPolyFit[i, :], temp = scipy.optimize.curve_fit(maths.polyVal, x, t, [0] * 6)
        VFuelPolyFit[i, :], temp = scipy.optimize.curve_fit(maths.polyVal, x, f, [0] * 6)

        if BPlot:  # TODO: save these plots in a subdirectory
            plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
            plt.tight_layout()
            plt.title('Lap Time Loss - Lift Zone ' + str(NZone[i]))
            plt.xlabel('rLift [-]')
            plt.ylabel('dtLap [s]')
            plt.scatter(rLift, tLapRLift[i, :])
            plt.plot(rLiftPlot, maths.polyVal(rLiftPlot, tLapPolyFit[i, 0], tLapPolyFit[i, 1], tLapPolyFit[i, 2], tLapPolyFit[i, 3], tLapPolyFit[i, 4], tLapPolyFit[i, 5]))
            plt.grid()
            plt.savefig(resultsDirPath + '/timeLoss_LiftZone_' + str(NZone[i]) + '.png', dpi=300, orientation='landscape', progressive=True)
            plt.close()

            plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
            plt.tight_layout()
            plt.title('Fuel Save - Lift Zone ' + str(NZone[i]))
            plt.xlabel('rLift [-]')
            plt.ylabel('dVFuel [l]')
            plt.scatter(rLift, VFuelRLift[i, :])
            plt.plot(rLiftPlot, maths.polyVal(rLiftPlot, VFuelPolyFit[i, 0], VFuelPolyFit[i, 1], VFuelPolyFit[i, 2], VFuelPolyFit[i, 3], VFuelPolyFit[i, 4], VFuelPolyFit[i, 5]))
            plt.grid()
            plt.savefig(resultsDirPath + '/fuelSave_LiftZone_' + str(NZone[i]) + '.png', dpi=300, orientation='landscape', progressive=True)
            plt.close()

    # maximum lift
    tLapMaxSave, VFuelMaxSave, R = costFcn(np.ones(len(NLiftEarliest)), car, c, NLiftEarliest, NBrake, None, False, LiftGear)

    # optimisation for 100 steps between maximum lift and push
    VFuelTGT = np.linspace(VFuelMaxSave, c['VFuel'][-1], 100)

    print('\n\tMaximum Lift:')
    print('\tLapTime:\t{} s'.format(np.round(tLapMaxSave, 3)))
    print('\tVFuel:\t\t{} l'.format(np.round(VFuelMaxSave, 3)))

    # bounds and constaints

    print('\n\tOptimising Fuel Saving...')
    bounds = [(0, 1)] * len(NLiftEarliest)
    LiftPointsVsFuelCons = {'VFuelTGT': np.empty((len(VFuelTGT), 1)), 'LiftPoints': np.empty((len(VFuelTGT), len(NLiftEarliest)))}

    result = []
    fun = []

    for i in range(0, len(VFuelTGT)):  # optimisation loop

        VFuelConsTGT = VFuelTGT[i] - c['VFuel'][-1]

        FuelConstraint = {'type': 'eq', 'fun': calcFuelConstraint, 'args': (VFuelPolyFit, VFuelConsTGT)}

        # actual optimisation
        temp_result = scipy.optimize.minimize(objectiveLapTime, np.zeros(len(NLiftEarliest)), args=(tLapPolyFit, VFuelPolyFit), method='SLSQP', bounds=bounds, constraints=FuelConstraint,
                                              options={'maxiter': 10000, 'ftol': 1e-09, 'iprint': 1, 'disp': False})

        result.append(temp_result)
        fun.append(temp_result['fun'])

        LiftPointsVsFuelCons['LiftPoints'][i, :] = result[i]['x']

    LiftPointsVsFuelCons['VFuelTGT'] = VFuelTGT
    LiftPointsVsFuelCons['tLapDelta'] = fun

    tLapVFuelPolyFit, _ = scipy.optimize.curve_fit(maths.polyVal, LiftPointsVsFuelCons['VFuelTGT'], fun, [0] * 6)

    plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    plt.tight_layout()
    plt.title('Delta tLap vs VFuel')
    plt.xlabel('VFuel [l]')
    plt.ylabel('Delta tLap [s]')
    plt.plot(LiftPointsVsFuelCons['VFuelTGT'], fun, label='Simulation')
    plt.plot(LiftPointsVsFuelCons['VFuelTGT'], maths.polyVal(LiftPointsVsFuelCons['VFuelTGT'], tLapVFuelPolyFit), label='PolyFit')
    plt.grid()
    plt.legend()
    plt.savefig(resultsDirPath + '/DetlatLap_vs_VFuel.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    plt.tight_layout()
    plt.title('rLift vs VFuelTGT')
    plt.xlabel('VFuelTGT [l]')
    plt.ylabel('rLift [-]')
    for k in range(0, len(NLiftEarliest)):
        plt.plot(LiftPointsVsFuelCons['VFuelTGT'], LiftPointsVsFuelCons['LiftPoints'][:, k], label='Lift Zone ' + str(NZone[k]))

    plt.legend()
    plt.grid()
    plt.savefig(resultsDirPath + '/rLift_vs_vFuelTGT.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    print('\t\tOptimisation complete')

    # different Lift Levels
    print('\n\tCalculating Lift Points and Consumption')
    NTempPlot = [0, 25, 75, 90, 95, 97, 99]
    # NTempPlot = range(35, 41)

    plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    plt.tight_layout()
    # plt.plot(c['LapDistPct'], c['VFuel'], label='Push')

    LiftPointsVsFuelCons['VFuelLift'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))
    LiftPointsVsFuelCons['VFuelReference'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))
    LiftPointsVsFuelCons['VFuelBudget'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))
    LiftPointsVsFuelCons['LapDistPctLift'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))

    corr = {'NLiftEarliest': NLiftEarliest, 'NBrake': NBrake, 'NReference': NReference, 'LiftGear': LiftGear, 'NApex': NApex}
    checkVFuel = np.array([])
    checkVFuelBudget = np.array([])
    VFuelOffThrottle = np.array([])
    VFuelOnThrottle = np.array([])

    print(NLiftEarliest)
    print(NBrake)
    
    # Lift points/sections: 0 = approaching first corner

    # NSplit = len(d['vCar']) - NCut - 1
    for i in range(0, len(VFuelTGT)):
        _, _, R = costFcn(LiftPointsVsFuelCons['LiftPoints'][i], car, c, NLiftEarliest, NBrake, None, False, LiftGear, NReference)

        VFuelTGT[i] = (R['VFuel'][-1] - R['VFuel'][0])
        VFuelCut2End = R['VFuel'][-1] - R['VFuel'][NCut]
        VFuelStart2Cut = R['VFuel'][NCut] - R['VFuel'][0]

        # VFuel at lift points
        tempVFuelLift = R['VFuel'][R['NLift']]
        tempVFuelLift[:-1] = tempVFuelLift[:-1] + VFuelCut2End
        tempVFuelLift[-1] = tempVFuelLift[-1] - R['VFuel'][NCut]
        LiftPointsVsFuelCons['VFuelLift'][i] = np.roll(tempVFuelLift, 1)

        # VFuel Budget
        # VFuelReference_temp = R['VFuelReference'][1:]
        # VFuelReference_temp = np.append(VFuelReference_temp, [R['VFuel'][-1] + R['VFuelReference'][0]])
        # LiftPointsVsFuelCons['VFuelBudget'][i] = np.roll(VFuelReference_temp - R['VFuelLift'], 2)
        tempVFuelBudget = np.roll(R['VFuel'][R['NLift']], -1) - R['VFuelReference']
        tempVFuelBudget[-1] = VFuelTGT[i] - R['VFuelReference'][-1] + R['VFuel'][R['NLift']][0]
        LiftPointsVsFuelCons['VFuelBudget'][i] = np.roll(tempVFuelBudget, 2)

        # fuel consumed off throttle
        VFuelOnThrottle = np.append(VFuelOnThrottle, np.sum(LiftPointsVsFuelCons['VFuelBudget'][i]))
        VFuelOffThrottle = np.append(VFuelOffThrottle, np.sum(R['VFuelReference'] - R['VFuel'][R['NLift']]))

        # VFuel at reference points
        tempVFuelReference = R['VFuel'][NReference]
        tempVFuelReference[:-1] = tempVFuelReference[:-1] + VFuelCut2End
        tempVFuelReference[-1] = tempVFuelReference[-1] - R['VFuel'][NCut]
        LiftPointsVsFuelCons['VFuelReference'][i] = np.roll(tempVFuelReference, 1)

        # LapDistPct at lift points
        tempLapDistPctLift = R['LapDistPct'][R['NLift']]
        tempLapDistPctLift[:-1] = tempLapDistPctLift[:-1] + (R['LapDistPct'][-1] - R['LapDistPct'][NCut])
        tempLapDistPctLift[-1] = tempLapDistPctLift[-1] - R['LapDistPct'][NCut]
        LiftPointsVsFuelCons['LapDistPctLift'][i] = np.roll(tempLapDistPctLift, 1)

        checkVFuelBudget = np.append(checkVFuelBudget, VFuelTGT[i] - np.sum(LiftPointsVsFuelCons['VFuelBudget'][i]))

        if i in NTempPlot:
            plt.plot(R['LapDistPct'], R['vCar'], label=str(round(VFuelTGT[i], 2)))
        # plt.plot(R['LapDistPct'], R['vCar'], label=str(round(VFuelTGT[i], 2)))

        if i == 50:
            temp50 = copy.deepcopy(R)
            # plt.scatter(R['LapDistPct'][NSplit], R['vCar'][NSplit], label=str(round(VFuelTGT[i], 2)))
            # plt.plot(R['LapDisVFuelReference_temptPct'], R['VFuel'], label=str(round(VFuelTGT[i], 2)))
            # plt.scatter(R['LapDistPct'][R['NLift']], R['VFuel'][R['NLift']], label=str(round(VFuelTGT[i], 2)))
            # plt.scatter(R['LapDistPct'][R['NLift']], R['VFuel'][R['NLift']], label=str(round(VFuelTGT[i], 2)))

    plt.legend()
    plt.grid()
    plt.savefig(resultsDirPath + '/overview.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    # # get LapDistPct and VFuel a Lift Point
    # LiftPointsVsFuelCons['LapDistPctLift'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))
    # LiftPointsVsFuelCons['VFuelLift2'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))
    # for i in range(0, len(NBrake)):  # lift zones
    #     # flip because x data must be monotonically increasing
    #     x = np.flip(1 - (np.linspace(NLiftEarliest[i], NBrake[i], NBrake[i]-NLiftEarliest[i]+1) - NLiftEarliest[i]) / (NBrake[i]-NLiftEarliest[i]))
    #     y = np.flip(c['LapDistPct'][np.linspace(NLiftEarliest[i], NBrake[i], NBrake[i]-NLiftEarliest[i]+1, dtype='int32')])
    #     for k in range(0, len(LiftPointsVsFuelCons['VFuelTGT'])):  # TGT
    #         LiftPointsVsFuelCons['LapDistPctLift'][k, i] = np.interp(LiftPointsVsFuelCons['LiftPoints'][k, i], x, y)
    #         LiftPointsVsFuelCons['VFuelLift2'][k, i] = np.interp(LiftPointsVsFuelCons['LapDistPctLift'][k, i], d['LapDistPct'], d['VFuel'])

    # # get LapDistPct and VFuel a Apex Point
    # LiftPointsVsFuelCons['LapDistPctApex'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))
    # LiftPointsVsFuelCons['VFuelApex'] = np.empty(np.shape(LiftPointsVsFuelCons['LiftPoints']))
    # for i in range(0, len(NApex)):  # lift zones
    #     for k in range(0, len(LiftPointsVsFuelCons['VFuelTGT'])):  # TGT
    #         LiftPointsVsFuelCons['LapDistPctApex'][k, i] = c['LapDistPct'][NApex[i]]
    #         LiftPointsVsFuelCons['VFuelApex'][k, i] = d['VFuel'][NApex[i]]

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

    NReference = NReference + len(d['vCar']) - NCut - 1
    NReference[NReference > len(d['vCar'])] = NReference[NReference > len(d['vCar'])] - len(d['vCar']) + 1
    NReference = np.sort(NReference)

    d['NApex'] = NApex
    d['NBrake'] = NBrake
    d['NLiftEarliest'] = NLiftEarliest
    d['NWOT'] = NWOT
    d['NReference'] = NReference

    print('NApex: ', d['NApex'])
    print('LapDistPctApex: ', d['LapDistPct'][d['NApex']])

    plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    plt.tight_layout()
    plt.plot(d['LapDistPct'], d['vCar'], label='d original')
    plt.plot(d['LapDistPct'], d['vCar'], label='d new', linestyle='dashed')
    plt.scatter(d['LapDistPct'][NApex], d['vCar'][NApex], label='NApex', zorder=99)
    plt.scatter(d['LapDistPct'][NBrake], d['vCar'][NBrake], label='NBrake', zorder=99)
    plt.scatter(d['LapDistPct'][NLiftEarliest], d['vCar'][NLiftEarliest], label='NLiftEarliest', zorder=99)
    plt.scatter(d['LapDistPct'][NWOT], d['vCar'][NWOT], label='NWOT', zorder=99)
    plt.scatter(d['LapDistPct'][NReference], d['vCar'][NReference], label='NReference', zorder=99)
    plt.scatter(d['LapDistPct'][d['NApex']], d['vCar'][NApex], label='NApex c', marker='x', zorder=99)
    plt.scatter(d['LapDistPct'][d['NBrake']], d['vCar'][NBrake], label='NBrake c', marker='x', zorder=99)
    plt.scatter(d['LapDistPct'][d['NLiftEarliest']], d['vCar'][NLiftEarliest], label='NLiftEarliest d', marker='x', zorder=99)
    plt.scatter(d['LapDistPct'][d['NWOT']], d['vCar'][NWOT], label='NWOT d', marker='x', zorder=99)
    plt.scatter(d['LapDistPct'][d['NReference']], d['vCar'][NReference], label='NReference d', marker='x', zorder=99)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    plt.savefig(resultsDirPath + '/resultsCheck.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    plt.tight_layout()
    plt.plot(d['x'], d['y'], label='Track')
    plt.scatter(d['x'][NApex], d['y'][NApex], label='NApex', zorder=99)
    plt.scatter(d['x'][NBrake], d['y'][NBrake], label='NBrake', zorder=99)
    plt.scatter(d['x'][NLiftEarliest], d['y'][NLiftEarliest], label='NLiftEarliest', zorder=99)
    plt.scatter(d['x'][NWOT], d['y'][NWOT], label='NWOT', zorder=99)
    plt.legend()
    for i in range(0, len(NBrake)):
        plt.annotate(s='Zone {}'.format(i+1), xy=(d['x'][NBrake][i], d['y'][NBrake][i]),
                     xycoords='data', xytext=(-10, -10), textcoords='offset points',
                     horizontalalignment='right', verticalalignment='top')
    plt.grid()
    plt.axis('equal')
    plt.savefig(resultsDirPath + '/trackMap.png', dpi=300, orientation='landscape', progressive=True)
    plt.close()

    LiftPointsVsFuelCons['LapDistPctWOT'] = d['LapDistPct'][d['NWOT']] * 100
    LiftPointsVsFuelCons['LapDistPctApex'] = d['LapDistPct'][d['NApex']] * 100
    LiftPointsVsFuelCons['LapDistPctBrake'] = d['LapDistPct'][d['NBrake']] * 100
    LiftPointsVsFuelCons['LapDistPctReference'] = d['LapDistPct'][d['NReference']] * 100

    # LiftPointsVsFuelCons['LapDistPctLift'] = (LiftPointsVsFuelCons['LapDistPctLift'] + 1-c['LapDistPct'][NCut]) * 100
    # LiftPointsVsFuelCons['LapDistPctLift'][LiftPointsVsFuelCons['LapDistPctLift'] > 100] = LiftPointsVsFuelCons['LapDistPctLift'][LiftPointsVsFuelCons['LapDistPctLift'] > 100] - 100
    LiftPointsVsFuelCons['SetupName'] = d['DriverInfo']['DriverSetupName']
    LiftPointsVsFuelCons['CarSetup'] = d['CarSetup']
    LiftPointsVsFuelCons['ibtFileName'] = ibtPath
    LiftPointsVsFuelCons['tLapVFuelPolyFit'] = tLapVFuelPolyFit

    LiftPointsVsFuelCons['LapDistPctLift'] = LiftPointsVsFuelCons['LapDistPctLift'].transpose() * 100
    LiftPointsVsFuelCons['VFuelBudget'] = LiftPointsVsFuelCons['VFuelBudget'].transpose()
    LiftPointsVsFuelCons['VFuelLift'] = LiftPointsVsFuelCons['VFuelLift'].transpose()
    LiftPointsVsFuelCons['VFuelReference'] = LiftPointsVsFuelCons['VFuelReference'].transpose()
    LiftPointsVsFuelCons['LiftPoints'] = LiftPointsVsFuelCons['LiftPoints'].transpose()
    LiftPointsVsFuelCons['SFuelConfigCarName'] = carScreenNameShort
    LiftPointsVsFuelCons['SFuelConfigTrackName'] = TrackName

    # export data
    # saveJson(LiftPointsVsFuelCons, resultsDirPath)

    print(time.strftime("%H:%M:%S", time.localtime()) + ':\tCompleted Fuel Saving Optimisation!')


    # # correlation
    # VFuelCorrelationTgt = 1.70
    #
    # LiftPoints = np.array([])
    #
    # for i in range(0, len(LiftPointsVsFuelCons['LapDistPctLift'])):
    #     LiftPoints = np.append(LiftPoints, np.interp(VFuelCorrelationTgt, LiftPointsVsFuelCons['VFuelTGT'], LiftPointsVsFuelCons['LiftPoints'][i]))
    #
    # _, _, R = costFcn(LiftPoints, car, c, corr['NLiftEarliest'], corr['NBrake'], None, False, corr['LiftGear'], corr['NReference'])
    #
    # # create new data dict for cut lap --> c
    # temp = copy.deepcopy(R)
    # RR = {}
    # keys = list(temp.keys())
    # NCut = len(temp[keys[10]][NApex[0]:-1])
    #
    # for i in range(0, len(temp)):
    #     if keys[i] == 'tLap':
    #         RR[keys[i]] = temp[keys[i]][NCut:-1] - R['tLap'][NCut]
    #         RR[keys[i]] = np.append(RR[keys[i]], temp[keys[i]][0:NCut] + RR[keys[i]][-1] + temp['dt'][-1])
    #     elif keys[i] == 'sLap':
    #         RR[keys[i]] = temp[keys[i]][NCut:-1] - R['sLap'][NCut]
    #         RR[keys[i]] = np.append(RR[keys[i]], temp[keys[i]][0:NCut] + RR[keys[i]][-1] + temp['ds'][-1])
    #     elif keys[i] == 'LapDistPct':
    #         RR[keys[i]] = temp[keys[i]][NCut:-1] - R['LapDistPct'][NCut]
    #         RR[keys[i]] = np.append(RR[keys[i]], temp[keys[i]][0:NCut] + RR[keys[i]][-1] + temp['dLapDistPct'][-1])
    #     elif keys[i] == 'VFuel':
    #         RR[keys[i]] = temp[keys[i]][NCut:-1] - R['VFuel'][NCut]
    #         RR[keys[i]] = np.append(RR[keys[i]], temp[keys[i]][0:NCut] + RR[keys[i]][-1] + temp['QFuel'][-1] * temp['dt'][-1])
    #     else:
    #         if type(temp[keys[i]]) is dict:
    #             RR[keys[i]] = temp[keys[i]]
    #         else:
    #             RR[keys[i]] = temp[keys[i]][NCut:-1]
    #             RR[keys[i]] = np.append(RR[keys[i]], temp[keys[i]][0:NCut])
    #
    # # get ibt path
    # ibtPath = filedialog.askopenfilename(initialdir=TelemPath, title="Select IBT file",
    #                                      filetypes=(("IBT files", "*.ibt"), ("all files", "*.*")))
    #
    # if not ibtPath:
    #     print(time.strftime("%H:%M:%S", time.localtime()) + ':\tNo valid path to ibt file provided...aborting!')
    #     return
    #
    # # imoport ibt file
    # f, _ = importIBT.importIBT(ibtPath,
    #                            lap=27,
    #                            channels=['zTrack', 'LapDistPct', 'rThrottle', 'rBrake', 'QFuel', 'SessionTime', 'VelocityX', 'VelocityY', 'Yaw', 'Gear', 'FuelLevel'],
    #                            channelMapPath=dirPath + '/functionalities/libs/iRacingChannelMap.csv')  # TODO: check if data is sufficient
    #
    #
    # plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    # plt.tight_layout()
    # plt.plot(RR['sLap'], RR['vCar'], label='Simulation')
    # plt.plot(f['sLap'], f['vCar'], label='iRacing')
    # plt.legend()
    # plt.grid()
    # plt.savefig(resultsDirPath + '/Correlation_vCar.png', dpi=300, orientation='landscape', progressive=True)
    # plt.close()
    #
    #
    # plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    # plt.tight_layout()
    # plt.plot(RR['sLap'], RR['QFuel'], label='Simulation')
    # plt.plot(f['sLap'], f['QFuel'], label='iRacing')
    # plt.legend()
    # plt.grid()
    # plt.savefig(resultsDirPath + '/Correlation_QFuel.png', dpi=300, orientation='landscape', progressive=True)
    # plt.close()
    #
    #
    # plt.figure(figsize=[16, 9], dpi=300)  # TODO: make plot nice
    # plt.tight_layout()
    # plt.plot(RR['sLap'], RR['VFuel']-RR['VFuel'][0], label='Simulation')
    # plt.plot(f['sLap'], -f['VFuel']+f['VFuel'][0], label='iRacing')
    # plt.legend()
    # plt.grid()
    # plt.savefig(resultsDirPath + '/Correlation_VFuel.png', dpi=300, orientation='landscape', progressive=True)
    # plt.close()
    #
