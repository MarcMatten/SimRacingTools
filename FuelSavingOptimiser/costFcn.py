import numpy as np
import copy

temp = None
rhoFuel = 0.75

def costFcn(rLift, *args):
    C = args[0]  # data struct
    NLiftEarliest = args[1]  # index of ealiest lift points
    NBrake = args[2]  # index of braking points
    VFuelTgt = args[3]  # fuel target for optimistion mode
    optim = args[4]  # optimisation mode -> calc error?
    LiftGear = args[5]
    for i in range(0, len(NLiftEarliest)):
        C = stepFwds(C, NLiftEarliest[i] + int((1 - rLift[i]) * (NBrake[i] - NLiftEarliest[i])), LiftGear[i])

    C = calcFuel(C)
    C = calcLapTime(C)

    if optim:
        return C['tLap'][-1] + abs(VFuelTgt - C['VFuel'][-1])*1000
    else:
        return C['tLap'][-1], C['VFuel'][-1], C


def stepFwds(x, n, LiftGear):
    temp = copy.deepcopy(x)
    i = 0

    while temp['vCar'][n] < temp['vCar'][n+1]:
        vCar = temp['vCar'][n]
        gLong = getGLong(vCar, temp['aTrackIncline'][n], LiftGear)
        ds = temp['ds'][n]
        temp['dt'][n] = -vCar/gLong - np.sqrt(np.square(vCar / gLong) + 2*ds/gLong)
        temp['vCar'][n+1] = temp['vCar'][n] + gLong * temp['dt'][n]
        temp['QFuel'][n] = 0.58
        n = n+1
        i += 1

    return temp


def getGLong(v, aTrackIncline, LiftGear):
    # return -0.00054902*np.square(v) +0.005411 * v - 1.0253 - np.sin(aTrackIncline/180*np.pi)
    if LiftGear > 4:
        return -0.0010998*np.square(v) + 0.057576 * v - 2.0634 - np.sin(aTrackIncline/180*np.pi)
    else:  # 4
        return -0.0007912*np.square(v) + 0.0008683 * v - 0.41181 - np.sin(aTrackIncline/180*np.pi)


def calcFuel(x):
    x['mFuel'] = np.cumsum(np.append(0, x['dt'] * x['QFuel'][0:-1]))/1000
    x['VFuel'] = x['mFuel']/rhoFuel

    return x


def calcLapTime(x):
    x['tLap'] = np.cumsum(np.append(0, x['dt']))
    return x
