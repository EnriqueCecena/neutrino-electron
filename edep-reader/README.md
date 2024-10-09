# OUTDATED!! REWRITE NEEDED"!!

# Requirements
- [edep-sim](https://github.com/ClarkMcGrew/edep-sim)

# Installation

### Get the code

```console
$ git clone https://baltig.infn.it/vpia/edep-reader.git
```

### Build the binaries

```console
$ cd edep-reader
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=./../install ./..
$ make
$ make install
```

In the `bin` folder, there will be one executable:
- **EDEPReader** will load an edepsim file and construct a trajectories tree

### Setup

```console
$ source setup.sh
```

# Run

### EDEPReader
- The input file is currently hardcoded at line 18.
- It process a single event from the input file.
- It creates the tree and perform some checks.

```console
$ EDEPReader
```

# TO-DO
- Prepare the package so that it can be used with the find_package interface.

