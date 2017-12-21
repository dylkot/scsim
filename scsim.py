import pandas as pd
import numpy as np

class scsim:
    def __init__(self, ngenes=10000, ncells=100, seed=757578,
                 mean_rate=.3, mean_shape=.6, libloc=11, libscale=0.2,
                expoutprob=.05, expoutloc=4, expoutscale=0.5, ngroups=1, groupprob=None,
                diffexpprob=.1, diffexpdownprob=.5, diffexploc=.1, diffexpscale=.4,
                bcv_dispersion=.1, bcv_dof=60):
        self.ngenes = ngenes
        self.ncells = ncells
        self.seed = seed
        self.mean_rate = mean_rate
        self.mean_shape = mean_shape
        self.libloc = libloc
        self.libscale = libscale
        self.expoutprob = expoutprob
        self.expoutloc = expoutloc
        self.expoutscale = expoutscale
        self.ngroups = ngroups
        self.groupprob = groupprob
        self.diffexpprob = diffexpprob
        self.diffexpdownprob = diffexpdownprob
        self.diffexploc = diffexploc
        self.diffexpscale = diffexpscale
        self.bcv_dispersion = bcv_dispersion
        self.bcv_dof = bcv_dof

    def simulate(self):
        self.cellparams = self.get_cell_params()
        self.geneparams = self.get_gene_params()
        self.sim_group_DE()
        self.cellgenemean = self.get_cell_gene_means()
        self.adjust_means_bcv()
        self.simulate_counts()



    def simulate_counts(self):
        self.counts = pd.DataFrame(np.random.poisson(lam=self.updatedmean),
                                   index=self.cellnames, columns=self.genenames)


    def adjust_means_bcv(self):
        self.bcv = (self.bcv_dispersion + (1 / np.sqrt(self.cellgenemean.astype(float))))
        chisamp = np.random.chisquare(self.bcv_dof, size=self.ngenes)
        self.bcv = self.bcv*np.sqrt(self.bcv_dof / chisamp)
        self.updatedmean = np.random.gamma(shape=1/(self.bcv**2),
                                           scale=self.cellgenemean*(self.bcv**2))
        self.bcv = pd.DataFrame(self.bcv, index=self.cellnames, columns=self.genenames)
        self.updatedmean = pd.DataFrame(self.updatedmean, index=self.cellnames,
                                        columns=self.genenames)

    def get_cell_gene_means(self):
        ind = self.cellparams['group'].apply(lambda x: 'group%d_genemean' % x)
        cellgenemean = self.geneparams.loc[:,ind].T
        cellgenemean.index = self.cellparams.index
        normfac = self.cellparams['libsize'] / cellgenemean.sum(axis=1)
        cellgenemean = cellgenemean.multiply(normfac, axis=0).astype(float)
        return(cellgenemean)


    def get_gene_params(self):
        basegenemean = np.random.gamma(shape=self.mean_shape,
                                       scale=1./self.mean_rate,
                                       size=self.ngenes)

        is_outlier = np.random.choice([True, False], size=self.ngenes,
                                      p=[self.expoutprob,1-self.expoutprob])
        outlier_ratio = np.ones(shape=self.ngenes)
        outliers = np.random.lognormal(mean=self.expoutloc, sigma=self.expoutscale,
                                      size=is_outlier.sum())
        outlier_ratio[is_outlier] = outliers
        gene_mean = basegenemean.copy()
        median = np.median(basegenemean)
        gene_mean[is_outlier] = outliers*median


        self.genenames = ['Gene%d' % i for i in range(1, self.ngenes+1)]
        geneparams = pd.DataFrame([basegenemean, is_outlier, outlier_ratio, gene_mean],
                                  index=['BaseGeneMean', 'is_outlier', 'outlier_ratio', 'gene_mean'],
                                 columns=self.genenames).T
        return(geneparams)


    def get_cell_params(self):
        groupid = self.simulate_groups()
        libsize = np.random.lognormal(mean=self.libloc, sigma=self.libscale,
                                      size=self.ncells)
        self.cellnames = ['Cell%d' % i for i in range(1, self.ncells+1)]
        cellparams = pd.DataFrame([groupid, libsize], index=['group', 'libsize'],
                                 columns=self.cellnames).T
        cellparams['group'] = cellparams['group'].astype(int)
        return(cellparams)


    def simulate_groups(self):
        if self.groupprob is None:
            self.groupprob = [1/float(self.ngroups)]*self.ngroups

        groupid = np.random.choice(np.arange(1, self.ngroups+1),
                                       size=self.ncells, p=self.groupprob)
        self.groups = np.unique(groupid)
        return(groupid)

    def sim_group_DE(self):
        groups = self.cellparams['group'].unique()
        for group in self.groups:
            isDE = np.random.choice([True, False], size=self.ngenes,
                                      p=[self.diffexpprob,1-self.diffexpprob])
            DEratio = np.random.lognormal(mean=self.diffexploc, sigma=self.diffexpscale,
                                      size=isDE.sum())
            DEratio[DEratio<1] = 1 / DEratio[DEratio<1]
            is_downregulated = np.random.choice([True, False], size=len(DEratio),
                                      p=[self.diffexpdownprob,1-self.diffexpdownprob])
            DEratio[is_downregulated] = 1. / DEratio[is_downregulated]
            all_DE_ratio = np.ones(self.ngenes)
            all_DE_ratio[isDE] = DEratio
            group_mean = self.geneparams['gene_mean']*all_DE_ratio

            deratiocol = 'group%d_DEratio' % group
            groupmeancol = 'group%d_genemean' % group
            self.geneparams[deratiocol] = all_DE_ratio
            self.geneparams[groupmeancol] = group_mean
