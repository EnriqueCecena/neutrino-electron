# Virtual environment to analyse the hdf5 files

This document provides the necessary instructions to set up the software required for analyzing `.hdf5` files.

## Setup

**Setting up the SL7 environment**  
  -  Before connecting to the DUNE computers via SSH, set up the SL7 environment with the following command:

```console
$ /cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer shell --shell=/bin/bash \
-B /cvmfs,/exp,/nashome,/pnfs/dune,/opt,/run/user,/etc/hostname,/etc/hosts,/etc/krb5.conf --ipc --pid \
/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest
```
  -  Export the UPS environment:

```console
$ export UPS_OVERRIDE="-H Linux64bit+3.10-2.17"
 ```
an then set up dune

```console
$ source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
```

  -  Move to the directory where you want to store your Python analysis scripts. For example:

```console
$ cd /exp/dune/app/users/tsantana/workdir
```
**Download the scripts required for the analysis**

```console
$ wget https://raw.githubusercontent.com/DUNE/2x2_sim/develop/run-validation/edepsim_validation.py
$ wget https://raw.githubusercontent.com/DUNE/2x2_sim/develop/run-validation/validation_utils.py
```
  -  Make a python virtual environment into which we can install python packages via the `pip` command:

```console
$ python3 -m venv analyseedep.venv
```

  -  Activate the virtual environment through:

```console
$ source analyseedep.venv/bin/activate
```

Note that when we start a new session on the gpvms at a later time and we want to run our analysis script, we have to run this command above again.


 - Install the required Python packages in the virtual environment

```console
$ pip install --upgrade pip setuptools wheel
$ pip install h5py matplotlib numpy awkward
```
 -  That’s it! At this point we should be able to run our analysis script using a command similar to the following:

```console
$ python name_of_analysis_script.py --sim_file /pnfs/dune/persistent/users/abooth/nd-production/MicroProdN1p2/output/run-convert2h5/MicroProdN1p2_NDLAr_1E18_RHC.convert2h5.nu/EDEPSIM_H5/0000000/0000100/MicroProdN1p2_NDLAr_1E18_RHC.convert2h5.nu.0000128.EDEPSIM.hdf5
```
Where name_of_analysis_script.py is whatever we called our analysis script and you can pass whatever .hdf5 file we want

### Note on HDF5 Files
 -  We are using HDF5 files located in the following path:
```console
$ /pnfs/dune/persistent/users/abooth/nd-production/MicroProdN1p2/output/run-convert2h5/MicroProdN1p2_NDLAr_1E18_RHC.convert2h5.nu/EDEPSIM_H5/0000000
```