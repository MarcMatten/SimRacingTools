import os
import time
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal

from functionalities.libs import filters, maths, importIBT
from libs.Car import Car


def getShiftRPM(dirPath):

    tReaction = 0.25  # TODO: as input to tune from GUI

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(initialdir=dirPath, title="Select IBT file",
                                           filetypes=(("IBT files", "*.ibt"), ("all files", "*.*")))
    
    carPath = filedialog.askopenfilename(initialdir=dirPath+"/data/car", title="Select car JSON file",
                                           filetypes=(("JSON files", "*.json"), ("all files", "*.*")))
    car = Car('CarName')
    car.load(carPath)

    print(time.strftime("%H:%M:%S", time.localtime()) + ':\tStarting Upshift calculation for: ' + car.name)

    d = importIBT.importIBT(path)

    # create results directory
    resultsDirPath = dirPath + "/data/shiftTone/" + car.name  # TODO: find better naming, e.g. based on car, track and data or comment
    if not os.path.exists(resultsDirPath):
        os.mkdir(resultsDirPath)

    d['BStraightLine'] = np.logical_and((d['gLat']) < 1, np.abs(d['SteeringWheelAngle']) < 0.03, np.abs(d['vCar']) > 10)
    d['BWOT'] = np.logical_and((d['rThrottle']) > 0.99, np.abs(d['rBrake']) < 0.01)
    d['BCoasting'] = np.logical_and((d['rThrottle']) < 0.01, np.abs(d['rBrake']) < 0.01)
    d['BShiftRPM'] = np.logical_and(d['BStraightLine'], d['BWOT'])
    d['BShiftRPM'] = np.logical_and(d['BShiftRPM'], d['gLong'] > 0.3)

    # minRPM = np.mean(d['RPM'][d['BShiftRPM']])
    minRPM = 2000

    plt.ioff()
    plt.figure(figsize=(8, 5))  # TODO: make plot nice
    plt.grid()
    plt.scatter(d['vCar'][d['BShiftRPM']], d['gLong'][d['BShiftRPM']])
    plt.xlabel('vCar [m/s]')
    plt.ylabel('gLong [m/s²]')
    plt.xlim(0, np.max(d['vCar'][d['BShiftRPM']]) + 5)
    plt.ylim(0, np.max(d['gLong'][d['BShiftRPM']]) + 1)

    d['BGear'] = list()
    d['BRPMRange'] = list()
    gLongPolyFit = list()
    RPMPolyFit = list()
    vCarMin = list()
    vCarMax = list()
    maxRPM = list()

    NGear = np.linspace(1, np.max(d['Gear']), np.max(d['Gear']))

    for i in range(0, np.max(d['Gear'])):

        d['BGear'].append(np.logical_and(d['BShiftRPM'], d['Gear'] == NGear[i]))

        maxRPM.append(np.max(d['RPM'][d['BGear'][i]]))

        tempBRPMRange = np.logical_and(d['BGear'][i], d['RPM'] > minRPM)
        tempBRPMRange = np.logical_and(tempBRPMRange, d['RPM'] < maxRPM[i])
        tempBRPMRange = np.logical_and(tempBRPMRange, filters.movingAverage(d['EngineWarnings'], 6) < 1)

        d['BRPMRange'].append(tempBRPMRange)

        PolyFitTemp, temp = scipy.optimize.curve_fit(maths.poly3, d['vCar'][d['BRPMRange'][i]], d['gLong'][d['BRPMRange'][i]])
        gLongPolyFit.append(PolyFitTemp)

        PolyFitTemp, temp = scipy.optimize.curve_fit(maths.poly2, d['vCar'][d['BRPMRange'][i]], d['RPM'][d['BRPMRange'][i]])
        RPMPolyFit.append(PolyFitTemp)

        vCarMin.append(np.min(d['vCar'][d['BRPMRange'][i]]))
        vCarMax.append(np.max(d['vCar'][d['BRPMRange'][i]]))
        vCar = np.linspace(vCarMin[i] - 10, vCarMax[i] + 10, 100)

        plt.scatter(d['vCar'][d['BRPMRange'][i]], d['gLong'][d['BRPMRange'][i]])
        plt.plot(vCar, maths.poly3(vCar, gLongPolyFit[i][0], gLongPolyFit[i][1], gLongPolyFit[i][2], gLongPolyFit[i][3]))

    vCarShiftOptimal = []
    vCarShiftTarget = []

    for k in range(0, np.max(d['Gear']) - 1):
        f1 = lambda x: maths.poly3(x, gLongPolyFit[k][0], gLongPolyFit[k][1], gLongPolyFit[k][2], gLongPolyFit[k][3])
        f2 = lambda x: maths.poly3(x, gLongPolyFit[k + 1][0], gLongPolyFit[k + 1][1], gLongPolyFit[k + 1][2], gLongPolyFit[k + 1][3])

        result = maths.findIntersection(f1, f2, vCarMax[k])

        vCarShiftOptimal.append(np.min([result[0], vCarMax[k]]))
        vCarShiftTarget.append(vCarShiftOptimal[k] - tReaction * maths.poly3(vCarShiftOptimal[k], gLongPolyFit[k][0], gLongPolyFit[k][1], gLongPolyFit[k][2], gLongPolyFit[k][3]))

        plt.scatter(vCarShiftOptimal[k], f1(vCarShiftOptimal[k]), marker='o', color='black')
        plt.scatter(vCarShiftTarget[k], f1(vCarShiftTarget[k]), marker='o', color='red')

    plt.savefig(resultsDirPath + '/gLong_vs_speed.png', dpi=300, orientation='landscape', progressive=True)

    plt.figure()  # TODO: make plot nice
    plt.scatter(d['vCar'][d['BShiftRPM']], d['RPM'][d['BShiftRPM']])
    plt.grid()
    plt.xlabel('vCar [m/s]')
    plt.ylabel('nMotor [RPM]')
    plt.xlim(0, np.max(d['vCar'][d['BShiftRPM']]) + 5)
    plt.ylim(0, np.max(d['RPM'][d['BShiftRPM']]) + 500)

    nMotorShiftOptimal = []
    nMotorShiftTarget = []

    for i in range(0, np.max(d['Gear'])):
        vCar = np.linspace(vCarMin[i] - 10, vCarMax[i] + 10, 100)
        plt.plot(vCar, maths.poly2(vCar, RPMPolyFit[i][0], RPMPolyFit[i][1], RPMPolyFit[i][2]))

        if i < np.max(d['Gear']) - 1:
            nMotorShiftOptimal.append(maths.poly2(vCarShiftOptimal[i], RPMPolyFit[i][0], RPMPolyFit[i][1], RPMPolyFit[i][2]))
            nMotorShiftTarget.append(maths.poly2(vCarShiftTarget[i], RPMPolyFit[i][0], RPMPolyFit[i][1], RPMPolyFit[i][2]))
            plt.scatter(vCarShiftOptimal[i], nMotorShiftOptimal[i], marker='o', color='black')
            plt.scatter(vCarShiftTarget[i], nMotorShiftTarget[i], marker='o', color='red')

    plt.savefig(resultsDirPath + '/RPM_vs_speed.png', dpi=300, orientation='landscape', progressive=True)

    # save so car file
    car.setShiftRPM(nMotorShiftOptimal, vCarShiftOptimal, nMotorShiftTarget, vCarShiftTarget, NGear[0:-1])
    car.save(carPath)

    print(time.strftime("%H:%M:%S", time.localtime()) + ':\tCompleted Upshift calculation!')

