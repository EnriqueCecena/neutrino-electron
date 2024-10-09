export UPS_OVERRIDE="-H Linux64bit+3.10-2.17"
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup geant4 v4_10_6_p01c -q e19:prof
setup gcc v8_2_0
setup git v2_20_1
setup python v2_7_13d -f Linux64bit+3.10-2.17
setup eigen v3_3_9a
setup root v6_18_04d -q e19:prof
setup cmake v3_19_6
setup genie v3_00_06h -q e19:prof
setup genie_xsec v3_00_04 -q G1810a0211a:e1000:k250
setup edepsim v3_2_0 -q e20:prof

