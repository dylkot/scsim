# scsim
Simulate single-cell RNA-SEQ data using the [Splatter](https://github.com/Oshlack/splatter) statistical framework, which is descrubed [here](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1305-0) but implemented in python. In addition, simulates doublet cells and cells with shared gene-expression programs. This was used to benchmark methods for gene expression program inference in single-cell rna-seq data as described [here](https://www.biorxiv.org/content/early/2018/10/07/310599)

The scsim class is contained in scsim.py and this contains the simulation code.

run_scsim.py has example code for running a simulation with a given set of parameters. It saves the results in the numpy compressed matrix format which can be loaded into a Pandas dataframe as follows:

    with np.load(filename) as f:
        result = pd.DataFrame(**f)

