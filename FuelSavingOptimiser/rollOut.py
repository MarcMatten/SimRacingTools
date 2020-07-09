import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt
from functionalities.libs import importIBT
from libs.Car import Car
import tkinter as tk
from tkinter import filedialog
import time
import os


def polyVal(x, *args):
    if isinstance(args[0], np.ndarray):
        c = args[0]
    else:
        c = args

    temp = 0

    for i in range(0, len(c)):
        temp += c[i] * np.power(x, i)

    return temp


def getRollOutCurve(dirPath):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(initialdir=dirPath, title="Select IBT file",
                                           filetypes=(("IBT files", "*.ibt"), ("all files", "*.*")))

    carPath = filedialog.askopenfilename(initialdir=dirPath+"/car", title="Select car JSON file",
                                           filetypes=(("JSON files", "*.json"), ("all files", "*.*")))
    car = Car('CarName')
    car.loadJson(carPath)

    print(time.strftime("%H:%M:%S", time.localtime()) + ':\tStarting roll-out curve calculation for: ' + car.name)

    d = importIBT.importIBT(path)

    # create results directory
    resultsDirPath = dirPath + "/FuelSaving/" + car.name
    os.mkdir(resultsDirPath)

    d['BStraightLine'] = np.logical_and(np.abs(d['gLat']) < 1, np.abs(d['SteeringWheelAngle']) < 0.03)
    d['BStraightLine'] = np.logical_and(d['BStraightLine'], d['vCar'] > 10)
    d['BCoasting'] = np.logical_and(d['rThrottle'] < 0.01, d['rBrake'] < 0.01)
    d['BCoasting'] = np.logical_and(d['BCoasting'], d['BStraightLine'])

    plt.ioff()
    plt.figure()
    plt.grid()
    plt.xlabel('vCar [m/s]')
    plt.ylabel('gLong [m/s²]')
    plt.xlim(0, np.max(d['vCar'][d['BCoasting']]) + 5)
    plt.ylim(np.min(d['gLong'][d['BCoasting']]) -1, 0)

    d['BGear'] = list()
    gLongPolyFit = list()
    QFuelPolyFit = list()

    NGear = np.linspace(0, np.max(d['Gear']), np.max(d['Gear'])+1)

    for i in range(0, np.max(d['Gear'])+1):
        d['BGear'].append(np.logical_and(d['BCoasting'], d['Gear'] == NGear[i]))

        PolyFitTemp, temp = scipy.optimize.curve_fit(polyVal, d['vCar'][d['BGear'][i]], d['gLong'][d['BGear'][i]], [0, 0, 0])
        gLongPolyFit.append(PolyFitTemp)

        PolyFitTemp, temp = scipy.optimize.curve_fit(polyVal, d['vCar'][d['BGear'][i]], d['QFuel'][d['BGear'][i]], [0, 0, 0])
        QFuelPolyFit.append(PolyFitTemp)

        plt.scatter(d['vCar'][d['BGear'][i]], d['gLong'][d['BGear'][i]])

        vCar = np.linspace(0, np.max(d['vCar'])+10, 100)
        plt.plot(vCar, polyVal(vCar, gLongPolyFit[i]))

    plt.savefig(resultsDirPath + '/roll_out_curve.png', dpi=300, orientation='landscape', progressive=True)

    plt.figure()
    plt.grid()
    plt.xlabel('vCar [m/s]')
    plt.ylabel('QFuel [g/s]')
    plt.xlim(0, np.max(d['vCar'][d['BCoasting']]) + 5)
    plt.ylim(0, np.max(d['QFuel'][d['BCoasting']]) + 10)

    for i in range(0, np.max(d['Gear']) + 1):
        plt.scatter(d['vCar'][d['BGear'][i]], d['QFuel'][d['BGear'][i]])
        plt.plot(vCar, polyVal(vCar, QFuelPolyFit[i]))

    plt.savefig(resultsDirPath + '/coastint_fuel_consumption.png', dpi=300, orientation='landscape', progressive=True)

    # save so car file
    car.setCoastingData(gLongPolyFit, QFuelPolyFit, NGear)
    car.saveJson(carPath)

    print(time.strftime("%H:%M:%S", time.localtime()) + ':\tCompleted roll-out calculation!')
