import argparse
import os, sys
from scsim import scsim
import numpy as np
import time



def save_df(obj, filename):
    '''Save pandas dataframe in compressed format'''
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)


def parse_args():
    parser = argparse.ArgumentParser(description='Run scsim with specified input arguments')
    parser.add_argument('--outdir', type=str, help='Output directory base')
    parser.add_argument('--seed', type=int,
                        help='seed for generating simulation seeds')
    parser.add_argument('--numsims', type=int, help='number of sims to run',
                        default=20)
    parser.add_argument('--firstseed', type=int, help='first seed to run',
                        default=0)
    parser.add_argument('--lastseed', type=int,
                        help='last seed to run,default is use all',
                        default=None)
    parser.add_argument('--deloc', type=float,
                        help='devalue',
                        default=1.)
    a = parser.parse_args()
    return(a.outdir, a.seed, a.numsims, a.firstseed, a.lastseed, a.deloc)





def main():
    (outdir, randseed, numsims, firstseed, lastseed, deval) = parse_args()
    ngenes=10000
    deloc=deval
    progdeloc=deval
    descale=1.0
    ndoublets=500
    K=13
    nproggenes = 400
    proggroups = [1,2,3,4]
    progcellfrac = .35
    ncells = 10000
    deprob = .025

    np.random.seed(randseed)
    simseeds = np.random.randint(1, high=2**15, size=numsims)
    if lastseed is None:
        lastseed = numsims+1


    for seed in simseeds[firstseed:lastseed]:
        print(seed)
        outbase = outdir % (seed, deloc, descale, ndoublets, progcellfrac, len(proggroups), nproggenes, deprob)
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        Koutbase = '%s/K%d' % (outbase, K)
        if not os.path.exists(Koutbase):
            os.mkdir(Koutbase)

        simulator = scsim(ngenes=ngenes, ncells=ncells, ngroups=K, libloc=7.64, libscale=0.78,
                 mean_rate=7.68,mean_shape=0.34, expoutprob=0.00286,
                 expoutloc=6.15, expoutscale=0.49,
                 diffexpprob=deprob, diffexpdownprob=0., diffexploc=deloc, diffexpscale=descale,
                 bcv_dispersion=0.448, bcv_dof=22.087, ndoublets=ndoublets,
                 nproggenes=nproggenes, progdownprob=0., progdeloc=progdeloc,
                 progdescale=descale, progcellfrac=progcellfrac, proggoups=proggroups,
                 minprogusage=.1, maxprogusage=.7, seed=seed)


        start_time = time.time()
        simulator.simulate()
        end_time = time.time()
        print('%.3f minutes elapsed for seed %d' % ((end_time-start_time)/60, seed))

        save_df(simulator.cellparams, '%s/cellparams' % Koutbase)
        save_df(simulator.geneparams, '%s/geneparams' % Koutbase)
        save_df(simulator.counts, '%s/counts' % Koutbase)


if __name__ == '__main__':
    main()
