# NEUTRINO ELECTRON 

This repository is for the use of the members of the neutrino-electron slack channel.

In here you will find a pre-compiled version of EDEPREADER and the EXAMPLES. This last directory has a file named "EDEPDisplay.cpp" which is the one that creates the images corresponding to a ".root" file obtained in an Energy Deposition Simulation (edep-sim).

## SETUP

In order to run the current program there's some stuff that needs to be prepared.

  -  Clone the repository.
```console
$ git clone https://github.com/EnriqueCecena/neutrino-electron
```
This will create a  ` neutrino -electron ` folder. Inside there's gonna be some shell setups that'll create an SL7 container to run the programm.
```console
$ cd neutrino-electron
$ source apptainer.sh
$ source main_setup.sh
 ```
  - Generate the build files (in case they aren't there)
```console
$ cd edep-reader
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=./../install ./..
$ make
$ make install
```  
There's no need to change the `CMakeLists.txt` file, it is already modified to the new version of CXX Compiler. 
  - Build the files to run the ` EDEPDisplay.cpp` file.
```console
$ cd ..
$ source setup.sh
$ cd install
$ source setup.sh
$ cd ..
$ cd examples
$ mkdir build
$ mkdir install
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=./../install ./..
```
  - Compile the program and run it.
```console
$ make
$ make install
$ cd bin
$ ./EDEPDisplay
```
### At the current stage it will generate a pair of energy and event images!!

With this setup, one can modify the EDEPDisplay.cpp and just compile it and run it inside the build folder.
