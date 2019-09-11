'''
Author: Jeff Adrion

'''

from ReLERNN.imports import *
from ReLERNN.helpers import *
from ReLERNN.sequenceBatchGenerator import *

class Manager(object):
    '''

    The manager class is a framework for handling both VCFs and masks
    and can multi-process many of the functions orginally found in ReLERNN_SIMULATE

    '''


    def __init__(self,
        vcf = None,
        chromosomes = None,
        mask = None,
        winSizeMx = None,
        vcfDir = None,
        projectDir = None,
        networkDir = None,
        phased = None,
        minS = None
        ):

        self.vcf = vcf
        self.chromosomes = chromosomes
        self.mask = mask
        self.winSizeMx = winSizeMx
        self.vcfDir = vcfDir
        self.projectDir = projectDir
        self.networkDir = networkDir
        self.phased = phased
        self.minS = minS


    def dumpGenotypes(self, wins=None, recombMap=None, trainValiTest=[0.8,0.1,0.1], nProc=1):
        '''
        writes genotype matrices for training, validation, and testing
        '''
        # read the map
        rMap = {}
        with open(recombMap, "r") as fIN:
            for line in fIN:
                ar=line.split()
                try:
                    rMap[ar[0]].append([int(ar[1]),int(ar[2]),float(ar[3])*1e-8])
                except KeyError:
                    rMap[ar[0]] = [[int(ar[1]),int(ar[2]),float(ar[3])*1e-8]]

        ### find the rate for each genomic window
        genomic_wins = []
        for win in wins:
            win_chrom = win[0]
            win_len = win[2]
            win_ct = win[6]
            start = 0
            for i in range(win_ct):
                genomic_wins.append([win_chrom, start, win_len])
                start += win_len

        # partition for multiprocessing
        print("\nCalculating a weighted recombination rate for each genomic window...")
        mpID = range(len(genomic_wins))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=genomic_wins, rMap

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_mapOverlap)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        rates = np.zeros(len(genomic_wins))
        for i in range(result_q.qsize()):
            item = result_q.get()
            rates[item[0]]=item[1]

        ## Write the npz files
        # partition for multiprocessing
        print("\nWriting the genotype and position files...")
        mpID = range(len(wins))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=self.vcfDir, self.vcf, self.chromosomes, self.projectDir, wins, trainValiTest

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_dumpGenotypes)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        ## Rename npz files and write the info.p
        for simSet in ["train","vali","test"]:
            fileIDs=[]
            simDir = os.path.join(self.projectDir, simSet)
            for path in glob.glob(os.path.join(simDir, "*_haps.npy")):
                info = os.path.basename(path).split("_")[0].split("-")
                fileIDs.append([int(info[0]),int(info[1])])

            infoP = {"genomic_win":[], "rho":[], "segSites":[], "numReps": len(fileIDs)}
            ct = 0
            for fID in sorted(fileIDs):
                infoP["genomic_win"].append(fID[0])
                infoP["segSites"].append(fID[1])
                infoP["rho"].append(rates[fID[0]])
                orgHapFILE = os.path.join(simDir,"{}-{}_haps.npy".format(fID[0],fID[1]))
                orgPosFILE = os.path.join(simDir,"{}-{}_pos.npy".format(fID[0],fID[1]))
                newHapFILE = os.path.join(simDir,"{}_haps.npy".format(ct))
                newPosFILE = os.path.join(simDir,"{}_pos.npy".format(ct))
                os.rename(orgHapFILE, newHapFILE)
                os.rename(orgPosFILE, newPosFILE)
                ct+=1
            infoP["genomic_win"] = np.array(infoP["genomic_win"])
            infoP["segSites"] = np.array(infoP["segSites"])
            infoP["rho"] = np.array(infoP["rho"])
            infofile = open(os.path.join(simDir,"info.p"),"wb")
            pickle.dump(infoP,infofile)
            infofile.close()
        return None


    def worker_mapOverlap(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                genomic_wins, rMap = params
                last_win = 0
                last_chrom = genomic_wins[0][0].split(":")[0]
                for i in mpID:
                    if genomic_wins[i][0].split(":")[0] != last_chrom:
                        last_win = 0
                        last_chrom = genomic_wins[i][0].split(":")[0]
                    M = mapOverlap(genomic_wins[i], last_win, rMap)
                    last_win = M[2]
                    if M[0] and sum(M[0]) > 0.0:
                        weighted_rate=np.average(np.array(M[1]), weights=np.array(M[0]))
                    else:
                        weighted_rate=None
                    result_q.put([i,weighted_rate])
            finally:
                task_q.task_done()


    def worker_dumpGenotypes(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                vcfDir, vcf, chroms, projectDir, wins, trainValiTest = params
                for i in mpID:
                    ## Read in the hdf5
                    bn=os.path.basename(self.vcf)
                    h5FILE=os.path.join(vcfDir,bn.replace(".vcf","_%s.hdf5" %(wins[i][0])))
                    callset=h5py.File(h5FILE, mode="r")
                    var=allel.VariantChunkedTable(callset["variants"],names=["CHROM","POS"], index="POS")
                    chroms=var["CHROM"]
                    pos=var["POS"]
                    genos=allel.GenotypeChunkedArray(callset["calldata"]["GT"])
                    winSize=wins[i][2]
                    batchSize=wins[i][4]

                    ## Set network parameters
                    bds_pred_params = {
                        'INFO':None,
                        'CHROM':chroms[0],
                        'WIN':winSize,
                        'IDs':get_index(pos,winSize),
                        'GT':genos,
                        'POS':pos,
                        'batchSize': batchSize,
                        'maxLen': None,
                        'frameWidth': 5,
                        'sortInds':False,
                        'center':False,
                        'ancVal':-1,
                        'padVal':0,
                        'derVal':1,
                        'realLinePos':True,
                        'posPadVal':0,
                        'phase':self.phased
                              }

                    ### Define sequence batch generator and get the window
                    vcf_window = VCFBatchGenerator(**bds_pred_params)
                    x,chrom,win,info,nSNPs = vcf_window.__getitem__(0)

                    ## Which genomic windows are on this chromosome?
                    assert len(x[0]) == wins[i][6], "Some windows missing genotypes"
                    start=0
                    for j in range(len(wins)):
                        if j < i:
                            start+=wins[j][6]
                    chrom_wins = range(start,start+wins[i][6])

                    ## Split the training validation and test sets and write the npz files
                    splitBool = np.random.choice([0,1,2], wins[i][6], p = trainValiTest)
                    for j in range(len(x[0])):
                        snp_thresh = self.minS
                        if x[0][j].shape[0] >= snp_thresh:
                            Hname = "{}-{}_haps.npy".format(chrom_wins[j],x[0][j].shape[0])
                            Pname = "{}-{}_pos.npy".format(chrom_wins[j],x[0][j].shape[0])
                            if splitBool[j] == 0:
                                direc = os.path.join(projectDir, "train")
                            elif splitBool[j] == 1:
                                direc = os.path.join(projectDir, "vali")
                            else:
                                direc = os.path.join(projectDir, "test")
                            Hpath = os.path.join(direc,Hname)
                            Ppath = os.path.join(direc,Pname)
                            np.save(Hpath,x[0][j])
                            np.save(Ppath,x[1][j])

            finally:
                task_q.task_done()


    def splitVCF(self,nProc=1):
        '''
        split the vcf into seperate files by chromosome
        '''
        # partition for multiprocessing
        mpID = range(len(self.chromosomes))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=self.vcfDir, self.vcf, self.chromosomes

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_splitVCF)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        return None


    def worker_splitVCF(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                vcfDir, vcf, chroms = params
                for i in mpID:
                    chrom = chroms[i].split(":")[0]
                    start = int(chroms[i].split(":")[1].split("-")[0])+1
                    end = int(chroms[i].split(":")[1].split("-")[1])+1
                    splitVCF=os.path.join(vcfDir, os.path.basename(vcf).replace(".vcf","_%s.vcf" %(chroms[i])))
                    print("Split chromosome: %s..." %(chrom))
                    with open(vcf, "r") as fIN, open(splitVCF, "w") as fOUT:
                        for line in fIN:
                            if line.startswith("#"):
                                fOUT.write(line)
                            if line.startswith("%s\t" %(chrom)):
                                pos = int(line.split()[1])
                                if start <= pos <= end:
                                    fOUT.write(line)
                    print("Converting %s to HDF5..." %(splitVCF))
                    h5FILE=splitVCF.replace(".vcf",".hdf5")
                    allel.vcf_to_hdf5(splitVCF,h5FILE,fields="*",overwrite=True)
                    os.system("rm %s" %(splitVCF))
            finally:
                task_q.task_done()


    def countSites(self, nProc=1):
        '''
        split the vcf into seperate files by chromosome
        '''
        # partition for multiprocessing
        mpID = range(len(self.chromosomes))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=self.chromosomes

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_countSites)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        wins = []
        for i in range(result_q.qsize()):
            item = result_q.get()
            wins.append(item)

        nSamps,maxS,maxLen = [],0,0
        sorted_wins = []
        winFILE=os.path.join(self.networkDir,"windowSizes.txt")
        with open(winFILE, "w") as fOUT:
            for chrom in self.chromosomes:
                for win in wins:
                    if win[0] == chrom:
                        maxS = max(maxS,win[4])
                        maxLen = max(maxLen,win[2])
                        nSamps.append(win[1])
                        sorted_wins.append(win)
                        fOUT.write("\t".join([str(x) for x in win])+"\n")
        if len(set(nSamps)) != 1:
            print("Error: chromosomes have different numbers of samples")
        return sorted_wins, nSamps[0], maxS, maxLen


    def worker_countSites(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                chromosomes = params
                for i in mpID:
                    h5FILE=os.path.join(self.vcfDir, os.path.basename(self.vcf).replace(".vcf","_%s.hdf5" %(chromosomes[i])))
                    print("""\nReading HDF5: "%s"...""" %(h5FILE))
                    callset=h5py.File(h5FILE, mode="r")
                    var=allel.VariantChunkedTable(callset["variants"],names=["CHROM","POS"], index="POS")
                    chroms=var["CHROM"]
                    pos=var["POS"]
                    genos=allel.GenotypeChunkedArray(callset["calldata"]["GT"])

                    #Is this a haploid or diploid VCF?
                    GT=genos.to_haplotypes()
                    GT=GT[:,1:2]
                    GT=GT[0].tolist()
                    if len(set(GT)) == 1 and GT[0] == -1:
                        nSamps=len(genos[0])
                    else:
                        nSamps=len(genos[0])*2

                    ## Identify ideal training parameters
                    step=1000
                    winSize=1000000
                    while winSize > 0:
                        ip = find_win_size(winSize,pos,step,self.winSizeMx)
                        if len(ip) != 5:
                            winSize-=step
                        else:
                            result_q.put([chromosomes[i],nSamps,ip[0],ip[1],ip[2],ip[3],ip[4]])
                            winSize=0
            finally:
                task_q.task_done()


    def maskWins(self, maxLen=None, wins=None, nProc=1):
        '''
        split the vcf into seperate files by chromosome
        '''
        ## Read accessability mask
        print("Accessibility mask found: calculating the proportion of the genome that is masked...")
        mask={}
        with open(self.mask, "r") as fIN:
            for line in fIN:
                ar = line.split()
                try:
                    mask[ar[0]].append([int(pos) for pos in ar[1:]])
                except KeyError:
                    mask[ar[0]] = [[int(pos) for pos in ar[1:]]]

        ## Combine genomic windows
        genomic_wins = []
        for win in wins:
            win_chrom = win[0]
            win_len = win[2]
            win_ct = win[6]
            start = 0
            for i in range(win_ct):
                genomic_wins.append([win_chrom, start, win_len])
                start += win_len

        # partition for multiprocessing
        mpID = range(len(genomic_wins))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=genomic_wins, mask, maxLen

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_maskWins)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        masks = []
        for i in range(result_q.qsize()):
            item = result_q.get()
            masks.append(item)

        mask_fraction, win_masks = [], []
        for mask in masks:
            mask_fraction.append(mask[0])
            win_masks.append(mask)

        mean_mask_fraction = sum(mask_fraction)/float(len(mask_fraction))
        print("{}% of genome inaccessible".format(round(mean_mask_fraction * 100,1)))
        return mean_mask_fraction, win_masks


    def worker_maskWins(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                genomic_wins, mask, maxLen = params
                last_win = 0
                last_chrom = genomic_wins[0][0].split(":")[0]
                for i in mpID:
                    if genomic_wins[i][0].split(":")[0] != last_chrom:
                        last_win = 0
                        last_chrom = genomic_wins[i][0].split(":")[0]
                    M = maskStats(genomic_wins[i], last_win, mask, maxLen)
                    last_win = M[2]
                    result_q.put(M)
            finally:
                task_q.task_done()

