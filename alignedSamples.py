def getAlignedSamples(dataPath, sharpName, flareName):
    
    flaringPaths = glob.glob('../shared/Data/HMI_LOS_SHARPS/valid_magnetograms/los/*_1.dat')
    nonFlaringPaths = glob.glob('../shared/Data/HMI_LOS_SHARPS/valid_magnetograms/los/*_0.dat')

    
    if not os.path.exists(flaringStorePath):
        os.makedirs(flaringStorePath)
    if not os.path.exists(nonflaringStorePath):
        os.makedirs(nonflaringStorePath)

    # Pre first flare (72 hours) for emerging flaring ARs
    print 'Processing Emerging ARs data'
    if not os.path.exists(flaringStorePath + '/emerging'):
        os.makedirs(flaringStorePath + '/emerging')
    TW = 72
    NOBS = TW*5
    for arNum in emerging_flaring_ARs:
        dates = getFlareDates(flareData[arNum])
        obS = sorted(sharpData[arNum].keys())
        firstTS = obS[0]
        lastTS = obS[-1]
        date = dates[0] 
        X = np.zeros((NOBS, Nfeat),dtype=np.float)
        counter = NOBS - 1
        while date >= firstTS and counter >= 0:
            if date in sharpData[arNum]:
                X[counter][:-1] = np.array(sharpData[arNum][date][:Nfeat-1])
                X[counter][-1] = 1.0
            counter -= 1
            date -= datetime.timedelta(seconds = 720)
        if X[:,Nfeat-1].sum() > 0.0: #Store if at least one valid observation is available
            X.dump(flaringStorePath + '/emerging/%d_0.dat'%(arNum))

    #collecting all observations from emerging nonflaring ARs         
    X = [] 
    for arNum in emerging_nonflaring_ARs:
        for date in sharpData[arNum]:
            X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(nonflaringStorePath + '/emerging.dat')

    print 'Processing Pre-Post Flaring ARs data'
    #PrePost Flare Observations from training and test data as well as quiet samples which are temporally separated from flares by at least 72 hours
    testStorePath = flaringStorePath + '/test'
    trainStorePath = flaringStorePath + '/train'
    if not os.path.exists(testStorePath):
        os.makedirs(testStorePath)
        os.makedirs(trainStorePath)

    X = [] #collecting all observations from training flaring ARs separated from flares by more than 72 hours 
    for arNum in flaring_ARs_train:
        dumpAlignedPrePostObs(arNum,sharpData[arNum],getFlareDates(flareData[arNum]),trainStorePath)
        quietObs = getFlareQuietObs(sharpData[arNum].keys(),getFlareDates(flareData[arNum]),endPoints[arNum][1])
        for date in quietObs:
            X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(flaringStorePath + '/quiet_train.dat')
 
    X = [] #collecting all observations from test flaring ARs separated from flares by more than 72 hours 
    for arNum in flaring_ARs_test:
        dumpAlignedPrePostObs(arNum,sharpData[arNum],getFlareDates(flareData[arNum]),testStorePath)
        quietObs = getFlareQuietObs(sharpData[arNum].keys(),getFlareDates(flareData[arNum]),endPoints[arNum][1])
        for date in quietObs:
            X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(flaringStorePath + '/quiet_test.dat')
                     
    print 'Processing  nonflaring ARs data'
    X = [] #collecting all observations from training nonflaring ARs
    #We consider observations from middle 72 hours of the observation span for a non-flaring AR.
    for arNum in nonflaring_ARs_train:
        obDates = sorted(sharpData[arNum].keys())
        mindate = obDates[len(obDates)/2] - datetime.timedelta(seconds=36*3600)
        maxdate = obDates[len(obDates)/2] + datetime.timedelta(seconds=36*3600)
        for date in sharpData[arNum]:
            if date <= maxdate and date >= mindate:
                X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(nonflaringStorePath + '/train.dat')

    X = [] #collecting all observations from test nonflaring ARs        
    for arNum in nonflaring_ARs_test:
        obDates = sorted(sharpData[arNum].keys())
        mindate = obDates[len(obDates)/2] - datetime.timedelta(seconds=36*3600)
        maxdate = obDates[len(obDates)/2] + datetime.timedelta(seconds=36*3600)
        for date in sharpData[arNum]:
            if date <= maxdate and date >= mindate:
                X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(nonflaringStorePath + '/test.dat')

    print 'Counting Samples'
    countSamples(dataPath) 
    print 'Done'
    return