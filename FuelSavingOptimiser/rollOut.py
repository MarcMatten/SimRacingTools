import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt
from importIBT import importIBT
import tkinter as tk
from tkinter import filedialog


def polyVal(x, *args):
    if isinstance(args[0], np.ndarray):
        c = args[0]
    else:
        c = args

    temp = 0

    for i in range(0, len(c)):
        temp += c[i] * np.power(x, i)

    return temp


def getRollOutCurve(path):
    # MyChannelMap = {'Speed': ['vCar', 1],  # m/s
    #                 'LapCurrentLapTime': ['tLap', 1],  # s
    #                 'LatAccel': ['gLat', 1],  # m/s²
    #                 'LongAccel': ['gLong', 1],  # m/s²
    #                 'ThrottleRaw': ['rThrottle', 1],  # 1
    #                 'BrakeRaw': ['rBrake', 1],  # 1
    #                 'FuelUsePerHour': ['QFuel', 1 / 3.6],  # l/h --> g/s
    #                 'LapDist': ['sLap', 1],  # m
    #                 'Alt': ['GPSAltitude', 1]  # m,
    #                 }

    d = importIBT(path)

    d['BStraightLine'] = np.logical_and(np.abs(d['gLat']) < 1, np.abs(d['SteeringWheelAngle']) < 0.03)
    d['BStraightLine'] = np.logical_and(d['BStraightLine'], d['vCar'] > 10)
    d['BCoasting'] = np.logical_and(d['rThrottle'] < 0.01, d['rBrake'] < 0.01)
    d['BCoasting'] = np.logical_and(d['BCoasting'], d['BStraightLine'])

    plt.figure()
    plt.grid()
    plt.xlabel('vCar [m/s]')
    plt.ylabel('gLong [m/s²]')
    # plt.legend()
    plt.show(block=False)
    plt.title('Title')
    # plt.xlim(0, np.max(d['vCar'][d['BShiftRPM']]) + 5)
    # plt.ylim(0, np.max(d['gLong'][d['BShiftRPM']]) + 1)

    d['BGear'] = list()
    gLongPolyFit = list()
    QFuelPolyFit = list()

    for i in range(0, np.max(d['Gear'])+1):
        NGear = i
        d['BGear'].append(np.logical_and(d['BCoasting'], d['Gear'] == NGear))

        # PolyFitTemp, temp = scipy.optimize.curve_fit(polyVal, d['vCar'][d['BGear'][i]], d['gLong'][d['BGear'][i]], [0, 0, 0, 0])
        PolyFitTemp, temp = scipy.optimize.curve_fit(np.polyval, d['vCar'][d['BGear'][i]], d['gLong'][d['BGear'][i]], [0, 0, 0, 0])
        gLongPolyFit.append(PolyFitTemp)

        # PolyFitTemp, temp = scipy.optimize.curve_fit(polyVal, d['vCar'][d['BGear'][i]], d['QFuel'][d['BGear'][i]], [0, 0, 0, 0])
        PolyFitTemp, temp = scipy.optimize.curve_fit(np.polyval, d['vCar'][d['BGear'][i]], d['QFuel'][d['BGear'][i]], [0, 0, 0, 0])
        QFuelPolyFit.append(PolyFitTemp)

        # plt.scatter(d['vCar'][d['BGear'][i]], d['gLong'][d['BGear'][i]])
        plt.scatter(d['vCar'][d['BGear'][i]], d['QFuel'][d['BGear'][i]])

        vCar = np.linspace(0, np.max(d['vCar'])+10, 100)
        # plt.plot(vCar, poly3(vCar, gLongPolyFit[i][0], gLongPolyFit[i][1], gLongPolyFit[i][2], gLongPolyFit[i][3]))
        # plt.plot(vCar, polyVal(vCar, gLongPolyFit[i]))
        # plt.plot(vCar, poly3(vCar, QFuelPolyFit[i][0], QFuelPolyFit[i][1], QFuelPolyFit[i][2], QFuelPolyFit[i][3]))
        # plt.plot(vCar, polyVal(vCar, QFuelPolyFit[i]))
        plt.plot(vCar, np.polyval(QFuelPolyFit[i]), vCar)

    return gLongPolyFit


# MyIbtPath = 'C:/Users/Marc/Documents/Projekte/SimRacingTools/FuelSavingOptimiser/fordgt2017_indianapolis oval 2020-05-11 19-43-16.ibt'
root = tk.Tk()
root.withdraw()
MyIbtPath = filedialog.askopenfilename(initialdir="C:/Users/Marc/Documents/Projekte/SimRacingTools/FuelSavingOptimiser", title="Select IBT file", filetypes=(("IBT files", "*.ibt"), ("all files", "*.*")))

p = getRollOutCurve(MyIbtPath)

print('Roll Out Curves: ', p)

print('Done')