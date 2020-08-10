import os
import time
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal

from functionalities.RTDB import RTDB
from functionalities.libs import maths, importIBT, importExport
from libs.Car import Car


def getRollOutCurve(dirPath):
    root = tk.Tk()
    root.withdraw()

    # get ibt path
    ibtPath = filedialog.askopenfilename(initialdir=dirPath, title="Select IBT file",
                                         filetypes=(("IBT files", "*.ibt"), ("all files", "*.*")))

    if not ibtPath:
        print(time.strftime("%H:%M:%S", time.localtime()) + ':\tNo valid path to ibt file provided...aborting!')
        return

    # imoport ibt file
    d, var_headers_names = importIBT.importIBT(ibtPath,
                                               channels=['zTrack', 'LapDistPct', 'rThrottle', 'rBrake', 'QFuel', 'RPM', 'SteeringWheelAngle', 'Gear', 'gLong', 'gLat', 'QFuel'],
                                               channelMapPath=dirPath + '/functionalities/libs/iRacingChannelMap.csv')

    setupName = d['DriverInfo']['DriverSetupName']
    DriverCarIdx = d['DriverInfo']['DriverCarIdx']
    carScreenNameShort = d['DriverInfo']['Drivers'][DriverCarIdx]['CarScreenNameShort']

    # If car file exists, load it. Otherwise, create new car object TODO: whole section is duplicate with getShiftRPM
    car = Car(carScreenNameShort)
    carFilePath = dirPath + '/data/car/' + carScreenNameShort + '.json'

    if carScreenNameShort + '.json' in importExport.getFiles(dirPath + '/data/car', 'json'):
        car.load(carFilePath)
    else:
        tempDB = RTDB.RTDB()
        tempDB.initialise(d, False)
        UserShiftRPM = [0] * 7
        UserShiftFlag = [False] * 7

        for k in range(0, np.max(d['Gear']) - 1):
            UserShiftRPM[k] = d['DriverInfo']['DriverCarSLShiftRPM']
            UserShiftFlag[k] = True

        tempDB.initialise({'UserShiftRPM': UserShiftRPM, 'UpshiftStrategy': 5, 'UserShiftFlag': UserShiftFlag}, False)

        car.createCar(tempDB, var_headers_names=var_headers_names)

        del tempDB

    print(time.strftime("%H:%M:%S", time.localtime()) + ':\tStarting roll-out curve calculation for: ' + car.name)

    # TODO: check it telemetry file is suitable

    # create results directory
    resultsDirPath = dirPath + "/data/fuelSaving/" + ibtPath.split('/')[-1].split('.ibt')[0]
    if not os.path.exists(resultsDirPath):
        os.mkdir(resultsDirPath)

    d['BStraightLine'] = np.logical_and(np.abs(d['gLat']) < 1, np.abs(d['SteeringWheelAngle']) < 10)
    d['BStraightLine'] = np.logical_and(d['BStraightLine'], d['vCar'] > 10)
    d['BCoasting'] = np.logical_and(d['rThrottle'] < 0.01, d['rBrake'] < 0.01)
    d['BCoasting'] = np.logical_and(d['BCoasting'], d['BStraightLine'])

    plt.ioff()
    plt.figure()  # TODO: make plot nice
    plt.grid()
    plt.xlabel('vCar [m/s]')
    plt.ylabel('gLong [m/s²]')
    plt.xlim(0, np.max(d['vCar'][d['BCoasting']]) + 5)
    plt.ylim(np.min(d['gLong'][d['BCoasting']])  * 1.1, 0)

    d['BGear'] = list()
    gLongPolyFit = list()
    QFuelPolyFit = list()
    vCar = np.linspace(0, np.max(d['vCar']) + 10, 100)
    NGear = np.linspace(0, np.max(d['Gear']), np.max(d['Gear'])+1)
    # NGear = np.linspace(1, np.max(d['Gear']), np.max(d['Gear']))

    for i in range(0, np.max(d['Gear'])+1):
    # for i in range(0, np.max(d['Gear'])):  # TODO: what if car can't coast in neutral?
        d['BGear'].append(np.logical_and(d['BCoasting'], d['Gear'] == NGear[i]))

        if i == 0:
            PolyFitgLong = [0, 0, 0]
            PolyFitQFuel = [0, 0, 0]
        else:
            PolyFitgLong, temp = scipy.optimize.curve_fit(maths.polyVal, d['vCar'][d['BGear'][i]], d['gLong'][d['BGear'][i]], [0, 0, 0])
            PolyFitQFuel, temp = scipy.optimize.curve_fit(maths.polyVal, d['vCar'][d['BGear'][i]], d['QFuel'][d['BGear'][i]], [0, 0, 0])

        gLongPolyFit.append(PolyFitgLong)
        QFuelPolyFit.append(PolyFitQFuel)

        if i > 0:
            plt.scatter(d['vCar'][d['BGear'][i]], d['gLong'][d['BGear'][i]])
            plt.plot(vCar, maths.polyVal(vCar, gLongPolyFit[i]))

    plt.savefig(resultsDirPath + '/roll_out_curve.png', dpi=300, orientation='landscape', progressive=True)

    plt.figure()  # TODO: make plot nice
    plt.grid()
    plt.xlabel('vCar [m/s]')
    plt.ylabel('QFuel [g/s]')
    plt.xlim(0, np.max(d['vCar'][d['BCoasting']]) + 5)
    plt.ylim(0, np.max(d['QFuel'][d['BCoasting']]) * 1.1)

    for i in range(0, np.max(d['Gear']) + 1):
        if i > 0:
            plt.scatter(d['vCar'][d['BGear'][i]], d['QFuel'][d['BGear'][i]])
            plt.plot(vCar, maths.polyVal(vCar, QFuelPolyFit[i]))

    plt.savefig(resultsDirPath + '/coasting_fuel_consumption.png', dpi=300, orientation='landscape', progressive=True)

    # save so car file
    car.setCoastingData(gLongPolyFit, QFuelPolyFit, NGear, setupName, d['CarSetup'])
    car.save(carFilePath)

    print(time.strftime("%H:%M:%S", time.localtime()) + ':\tCompleted roll-out calculation!')
