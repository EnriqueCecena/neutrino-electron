
#PDG ID  = 10LZZZAAAI

# 10    :   A prefix indicating the particle is a nucleus.
# L     :   Total number of strange quarks (usually 0 for nuclei without strange quarks).
# ZZZ   :   Atomic number  Z  (number of protons), zero-padded to three digits.
# I     :   Isomer number (state of excitation), where:  0  =  Ground state & 1, 2, 3, ... =  Excited (isomeric) states.

pdg_id_map = {
    # Leptons
    '-16'   : 'Anti-Tau_Neutrino',        # Anti-Tau Neutrino
    '-15'   : 'Tau+',                     # Tau Lepton (Positive)
    '-14'   : 'Anti-Muon_Neutrino',       # Anti-Muon Neutrino
    '-13'   : 'Muon+',                    # Muon (Positive)
    '-12'   : 'Anti-Electron_Neutrino',   # Anti-Electron Neutrino
    '-11'   : 'Positron',                 # Positron
    '11'    : 'Electron',                 # Electron
    '12'    : 'Electron_Neutrino',        # Electron Neutrino
    '13'    : 'Muon-',                    # Muon (Negative)
    '14'    : 'Muon_Neutrino',            # Muon Neutrino
    '15'    : 'Tau-',                     # Tau Lepton (Negative)
    '16'    : 'Tau_Neutrino',             # Tau Neutrino

    # Gauge Bosons
    '-24'   : 'W-',                       # W Boson Negative
    '21'    : 'Gluon',                    # Gluon
    '22'    : 'Photon',                   # Photon
    '23'    : 'Z0',                       # Z Boson
    '24'    : 'W+',                       # W Boson Positive

    # Quarks
    '-6'    : 'Anti-Top_Quark',           # Anti-Top Quark
    '-5'    : 'Anti-Bottom_Quark',        # Anti-Bottom Quark
    '-4'    : 'Anti-Charm_Quark',         # Anti-Charm Quark
    '-3'    : 'Anti-Strange_Quark',       # Anti-Strange Quark
    '-2'    : 'Anti-Up_Quark',            # Anti-Up Quark
    '-1'    : 'Anti-Down_Quark',          # Anti-Down Quark
    '1'     : 'Down_Quark',               # Down Quark
    '2'     : 'Up_Quark',                 # Up Quark
    '3'     : 'Strange_Quark',            # Strange Quark
    '4'     : 'Charm_Quark',              # Charm Quark
    '5'     : 'Bottom_Quark',             # Bottom Quark
    '6'     : 'Top_Quark',                # Top Quark

    # Mesons
    '310'   : 'K0_S-',                    # Kaon -Short
    '-541'  : 'Bc-',                      # Bc Meson Negative
    '-531'  : 'Bs0_Bar',                  # Anti-Bs0 Meson
    '-521'  : 'B-',                       # B Meson Negative
    '-511'  : 'B0_Bar',                   # Anti-B0 Meson
    '-431'  : 'Ds-',                      # Ds Meson Negative
    '-421'  : 'D0_Bar',                   # Anti-D0 Meson
    '-411'  : 'D-',                       # D Meson Negative
    '-321'  : 'K-',                       # Kaon Negative
    '-211'  : 'pi-',                      # Pion Negative
    '-311'  : 'K0_Bar',                   # Anti-Kaon Zero
    '111'   : 'pi0',                      # Pion Zero
    '211'   : 'pi+',                      # Pion Positive
    '221'   : 'eta',                      # Eta Meson
    '311'   : 'K0',                       # Kaon Zero
    '321'   : 'K+',                       # Kaon Positive
    '411'   : 'D+',                       # D Meson Positive
    '421'   : 'D0',                       # D Meson Zero
    '431'   : 'Ds+',                      # Ds Meson Positive
    '511'   : 'B0',                       # B Meson Zero
    '521'   : 'B+',                       # B Meson Positive
    '531'   : 'Bs0',                      # Bs Meson Zero
    '541'   : 'Bc+',                      # Bc Meson Positive
    '553'   : 'Upsilon(1S)',              # Upsilon Meson

    # Baryons
    '-3334' : 'Anti-Omega-',              # Anti-Omega Negative
    '-3322' : 'Anti-Xi0',                 # Anti-Xi Zero
    '-3312' : 'Anti-Xi-',                 # Anti-Xi Negative
    '-3222' : 'Anti-Sigma+',              # Anti-Sigma Positive
    '-3212' : 'Anti-Sigma0',              # Anti-Sigma Zero
    '-3122' : 'Anti-Lambda0',             # Anti-Lambda Baryon
    '-3112' : 'Anti-Sigma-',              # Sigma Negative 
    '-2112' : 'Anti-Neutron',             # Anti-Neutron
    '-2212' : 'Anti-Proton',              # Anti-Proton
    '2112'  : 'Neutron',                  # Neutron
    '2212'  : 'Proton',                   # Proton
    '3112'  : 'Sigma-',                   # Sigma Negative
    '3122'  : 'Lambda0',                  # Lambda Baryon
    '3212'  : 'Sigma0',                   # Sigma Zero
    '3222'  : 'Sigma+',                   # Sigma Positive
    '3312'  : 'Xi-',                      # Xi Negative
    '3322'  : 'Xi0',                      # Xi Zero
    '3334'  : 'Omega-',                   # Omega Negative,

    # Other Mesons
    '-545'  : 'Anti-B*_c2',               # Anti-B*_c2 Meson
    '-543'  : 'B*_c-',                    # B*_c Meson Negative
    '-535'  : 'Anti-B*_s2(5840)0',        # Anti-B*_s2(5840)0 Meson
    '-533'  : 'Anti-B*_s0',               # Anti-B*_s0 Meson
    '-525'  : 'B*_2(5747)-',              # B*_2(5747) Meson Negative
    '-523'  : 'B*-',                      # B* Meson Negative
    '-515'  : 'Anti-B*_2(5747)0',         # Anti-B*_2(5747)0 Meson
    '-513'  : 'Anti-B*0',                 # Anti-B*0 Meson
    '-435'  : 'D*_s2(2573)-',             # D*_s2(2573) Meson Negative
    '-433'  : 'D*_s-',                    # D*_s Meson Negative
    '-425'  : 'Anti-D*(2460)0',           # Anti-D*(2460)0 Meson
    '-423'  : 'Anti-D*(2007)0',           # Anti-D*(2007)0 Meson
    '-415'  : 'D*(2460)-',                # D*(2460) Meson Negative
    '-413'  : 'D*(2010)-',                # D*(2010) Meson Negative
    '-329'  : 'K*(2045)-',                # K*(2045) Meson Negative
    '-327'  : 'K*(1780)-',                # K*(1780) Meson Negative
    '-325'  : 'K*(1430)-',                # K*(1430) Meson Negative
    '-323'  : 'K*(892)-',                 # K*(892) Meson Negative
    '-319'  : 'Anti-K*(2045)0',           # Anti-K*(2045)0 Meson
    '-317'  : 'Anti-K*(1780)0',           # Anti-K*(1780)0 Meson
    '-315'  : 'Anti-K*(1430)0',           # Anti-K*(1430)0 Meson
    '-313'  : 'Anti-K*(892)0',            # Anti-K*(892)0 Meson
    '-219'  : 'a4(2040)-',                # a4(2040) Meson Negative
    '-217'  : 'rho3(1690)-',              # rho3(1690) Meson Negative
    '-215'  : 'a2(1320)-',                # a2(1320) Meson Negative
    '-213'  : 'rho(770)-',                # rho(770) Meson Negative
    '113'   : 'rho(770)0',                # rho(770) Meson Zero
    '115'   : 'a2(1320)0',                # a2(1320) Meson Zero
    '117'   : 'rho3(1690)0',              # rho3(1690) Meson Zero
    '119'   : 'a4(2040)0',                # a4(2040) Meson Zero
    '130'   : 'K0_L',                     # Kaon Zero Long
    '213'   : 'rho(770)+',                # rho(770) Meson Positive
    '215'   : 'a2(1320)+',                # a2(1320) Meson Positive
    '217'   : 'rho3(1690)+',              # rho3(1690) Meson Positive
    '219'   : 'a4(2040)+',                # a4(2040) Meson Positive
    '223'   : 'omega(782)',               # Omega Meson
    '225'   : 'f2(1270)',                 # f2(1270) Meson
    '227'   : 'omega3(1670)',             # omega3(1670) Meson
    '229'   : 'f4(2050)',                 # f4(2050) Meson
    '313'   : 'K*(892)0',                 # K*(892) Meson Zero
    '315'   : 'K*(1430)0',                # K*(1430) Meson Zero
    '317'   : 'K*(1780)0',                # K*(1780) Meson Zero
    '319'   : 'K*(2045)0',                # K*(2045) Meson Zero
    '323'   : 'K*(892)+',                 # K*(892) Meson Positive
    '325'   : 'K*(1430)+',                # K*(1430) Meson Positive
    '327'   : 'K*(1780)+',                # K*(1780) Meson Positive
    '329'   : 'K*(2045)+',                # K*(2045) Meson Positive
    '331'   : 'eta\'(958)',               # Eta Prime Meson
    '333'   : 'phi(1020)',                # Phi Meson
    '335'   : 'f\'2(1525)',               # f'2(1525) Meson
    '337'   : 'phi3(1850)',               # phi3(1850) Meson
    '413'   : 'D*(2010)+',                # D*(2010) Meson Positive
    '415'   : 'D*(2460)+',                # D*(2460) Meson Positive
    '423'   : 'D*(2007)0',                # D*(2007) Meson Zero
    '425'   : 'D*(2460)0',                # D*(2460) Meson Zero
    '433'   : 'D*_s+',                    # D*_s Meson Positive
    '435'   : 'D*_s2(2573)+',             # D*_s2(2573) Meson Positive
    '513'   : 'B*0',                      # B* Meson Zero
    '515'   : 'B*_2(5747)0',              # B*_2(5747) Meson Zero
    '523'   : 'B*+',                      # B* Meson Positive
    '525'   : 'B*_2(5747)+',              # B*_2(5747) Meson Positive
    '533'   : 'B*_s0',                    # B*_s Meson Zero
    '535'   : 'B*_s2(5840)0',             # B*_s2(5840) Meson Zero
    '543'   : 'B*_c+',                    # B*_c Meson Positive
    '545'   : 'B*_c2+',                   # B*_c2 Meson Positive
    '553'   : 'Upsilon(1S)',              # Upsilon Meson

    # Charmed Baryons
    '-4332' : 'Anti-Omega_c0',            # Anti-Omega_c0 Baryon
    '-4232' : 'Anti-Xi_c-',               # Anti-Xi_c- Baryon
    '-4222': 'Anti-Sigma_c++',           # Anti-Sigma_c++
    '-4132' : 'Anti-Xi_c0',               # Anti-Xi_c0 Baryon
    '-4122' : 'Anti-Lambda_c-',           # Anti-Lambda_c- Baryon

    # Hyperons
    '4112'  : 'Sigma_c0',                 # Sigma_c0 Baryon
    '4122'  : 'Lambda_c+',                # Lambda_c+ Baryon
    '4132'  : 'Xi_c0',                    # Xi_c0 Baryon
    '4212'  : 'Sigma_c+',                 # Sigma_c+ Baryon
    '4222'  : 'Sigma_c++',                # Sigma_c++ Baryon
    '4232'  : 'Xi_c+',                    # Xi_c+ Baryon
    '4312'  : "Xi'_c0",                   # Xi'_c0 Baryon
    '4322'  : "Xi'_c+",                   # Xi'_c+ Baryon
    '4332'  : 'Omega_c0',                 # Omega_c0 Baryon

    # Top and Bottom Baryons
    '-5332' : 'Anti-Omega_b+',            # Anti-Omega_b+ Baryon
    '-5232' : 'Anti-Xi_b0',               # Anti-Xi_b0 Baryon
    '-5132' : 'Anti-Xi_b+',               # Anti-Xi_b+ Baryon
    '-5122' : 'Anti-Lambda_b0',           # Anti-Lambda_b0 Baryon
    '5122'  : 'Lambda_b0',                # Lambda_b0 Baryon
    '5132'  : 'Xi_b-',                    # Xi_b- Baryon
    '5232'  : 'Xi_b0',                    # Xi_b0 Baryon
    '5332'  : 'Omega_b-',                 # Omega_b- Baryon

    # Scalar Mesons
    '9010221' : 'f0(980)',                # f0(980) Meson
    '9020221' : 'f0(1370)',               # f0(1370) Meson

    # Glueballs and Hybrids (Hypothetical)
    '990'     : 'Glueball',               # Glueball
    '1101'    : 'Hybrid_Meson',           # Hybrid Meson

    # Nuclei
    '1000010020': 'D',                     # Deuteron (Hydrogen-2)
    '1000010030': 'T',                     # Triton (Hydrogen-3)
    '1000010040': 'He-4',                  # Alpha Particle
    '1000020030': 'He-3',                  # Helium-3
    '1000020040': 'He-4',                  # Helium-4
    '1000020050': 'He-5',                  # Helium-6 
    '1000020060': 'He-6',                  # Helium-6
    '1000020070': 'He-7',                  # Helium-7 
    '1000020080': 'He-8',                  # Helium-8 
    '1000020090': 'He-9',                  # Helium-9 
    '1000020100': 'He-10',                 # Helium-10 
    '1000040060': 'Be-6',                  # Beryllium-6
    '1000040070': 'Be-7',                  # Beryllium-7
    '1000040080': 'Be-8',                  # Beryllium-8
    '1000040090': 'Be-9',                  # Beryllium-9
    '1000040100': 'Be-10',                 # Beryllium-10
    '1000040110': 'Be-11',                 # Beryllium-11
    '1000040120': 'Be-12',                 # Beryllium-12
    '1000050080': 'B-8',                   # Boron-8 
    '1000050090': 'B-9',                   # Boron-9
    '1000050100': 'B-10',                  # Boron-10
    '1000050109': 'B-10m',                 # Boron-10 (Metastable)
    '1000050110': 'B-11',                  # Boron-11 
    '1000050120': 'B-12',                  # Boron-12 
    '1000050130': 'B-13',                  # Boron-13 
    '1000050140': 'B-14',                  # Boron-14 
    '1000060100': 'C-10',                  # Carbon-10 
    '1000060110': 'C-11',                  # Carbon-11
    '1000060120': 'C-12',                  # Carbon-12
    '1000060140': 'C-14',                  # Carbon-14
    '1000060150': 'C-15',                  # Carbon-15 
    '1000060159': 'C-15m',                 # Carbon-15  (Metastable) 
    '1000060160': 'C-16',                  # Carbon-16
    '1000060130': 'C-13',                  # Carbon-13
    '1000030040': 'Li-4',                  # Litium-4 
    '1000030050': 'Li-5',                  # Litium-5 
    '1000030060': 'Li-6',                  # Litium-6 
    '1000030070': 'Li-7',                  # Litium-7 
    '1000030080': 'Li-8',                  # Litium-8 
    '1000030090': 'Li-9',                  # Litium-9
    '1000030100': 'Li-10',                 # Litium-10
    '1000070120': 'N-12',                  # Nitrogen-12
    '1000070130': 'N-13',                  # Nitrogen-13
    '1000070140': 'N-14',                  # Nitrogen-14
    '1000070150': 'N-15',                  # Nitrogen-15
    '1000070160': 'N-16',                  # Nitrogen-16
    '1000070161': 'N-16m',                 # Nitrogen-16 (Metastable)
    '1000070169': 'N-16m',                 # Nitrogen-16 (Metastable)
    '1000070170': 'N-17',                  # Nitrogen-17
    '1000070180': 'N-18',                  # Nitrogen-18
    '1000070190': 'N-19',                  # Nitrogen-19
    '1000070200': 'N-20',                  # Nitrogen-20
    '1000090170': 'F-17',                  # Fluorine-17 
    '1000090180': 'F-18',                  # Fluorine-18 
    '1000090189': 'F-18m',                 # Fluorine-18 (Metastable)
    '1000090190': 'F-19',                  # Fluorine-19 
    '1000090199': 'F-19m',                 # Fluorine-19 (Metastable)
    '1000090200': 'F-20',                  # Fluorine-20 
    '1000090210': 'F-21',                  # Fluorine-21 
    '1000090219': 'F-21m',                 # Fluorine-21 (Metastable) 
    '1000090220': 'F-22',                  # Fluorine-22 
    '1000090230': 'F-23',                  # Fluorine-23
    '1000080140': 'O-14',                  # Oxygen-14
    '1000080150': 'O-15',                  # Oxygen-15
    '1000080160': 'O-16',                  # Oxygen-16
    '1000080170': 'O-17',                  # Oxygen-17
    '1000080180': 'O-18',                  # Oxygen-18
    '1000080190': 'O-19',                  # Oxygen-19 
    '1000080199': 'O-19m',                 # Oxygen-19 (Metastable)
    '1000080200': 'O-20',                  # Oxygen-20
    '1000080210': 'O-21',                  # Oxygen-21
    '1000080220': 'O-22m',                 # Oxygen-22 (Metastable)
    '1000090240': 'F-24' ,                 # Flourine-24 
    '1000100180': 'Ne-18',                 # Neon-18  
    '1000100190': 'Ne-19',                 # Neon-19   
    '1000100200': 'Ne-20',                 # Neon-20
    '1000100210': 'Ne-21',                 # Neon-21
    '1000100220': 'Ne-22',                 # Neon-22
    '1000100230': 'Ne-23',                 # Neon-23         
    '1000100240': 'Ne-24',                 # Neon-24            
    '1000100249': 'Ne-24m',                # Neon-24 (Metastable)                
    '1000100250': 'Ne-25',                 # Neon-25                 
    '1000100260': 'Ne-26',                 # Neon-26                 
    '1000100270': 'Ne-27',                 # Neon-27                 
    '1000110200': 'Na-20',                 # Sodium-20
    '1000110210': 'Na-21',                 # Sodium-21
    '1000110220': 'Na-22',                 # Sodium-22
    '1000110229': 'Na-22m',                # Sodium-22 (Metastable)
    '1000110230': 'Na-23',                 # Sodium-23
    '1000110240': 'Na-24',                 # Sodium-24
    '1000110241': 'Na-24m',                # Sodium-24 (Metastable)
    '1000110249': 'Na-24m',                # Sodium-24 (Metastable)
    '1000110250': 'Na-25',                 # Sodium-25
    '1000110259': 'Na-25m',                # Sodium-25 (Metastable)
    '1000110260': 'Na-26',                 # Sodium-26
    '1000110261': 'Na-26m',                # Sodium-26 (Metastable)
    '1000110270': 'Na-27',                 # Sodium-27
    '1000110280': 'Na-28',                 # Sodium-28
    '1000110290': 'Na-29',                 # Sodium-29
    '1000120210': 'Mg-21',                 # Magnesium-21
    '1000120220': 'Mg-22',                 # Magnesium-22
    '1000120230': 'Mg-23',                 # Magnesium-23
    '1000120239': 'Mg-23m',                # Magnesium-23 (Metastable)
    '1000120240': 'Mg-24m',                # Magnesium-24 (Metastable)
    '1000120250': 'Mg-25',                 # Magnesium-25
    '1000120259': 'Mg-25m',                # Magnesium-25 (Metastable)
    '1000120260': 'Mg-26',                 # Magnesium-26
    '1000120270': 'Mg-27',                 # Magnesium-27
    '1000120280': 'Mg-28',                 # Magnesium-28
    '1000120290': 'Mg-29',                 # Magnesium-29
    '1000120299': 'Mg-29m',                # Magnesium-29 (Metastable)
    '1000120300': 'Mg-30',                 # Magnesium-30
    '1000120310': 'Mg-31',                 # Magnesium-31
    '1000120320': 'Mg-32',                 # Magnesium-32
    '1000120330': 'Mg-33',                 # Magnesium-33
    '1000130241': 'Al-24m',                # Aluminum-24 (Metastable)
    '1000130250': 'Al-25',                 # Aluminum-25
    '1000130259': 'Al-25m',                # Aluminum-25 (Metastable)
    '1000130260': 'Al-26',                 # Aluminum-26 
    '1000130269': 'Al-26m',                # Aluminum-26 (Metastable)
    '1000130270': 'Al-27',                 # Aluminum-27
    '1000130280': 'Al-28',                 # Aluminum-28
    '1000130289': 'Al-28m',                # Aluminum-28 (Metastable)
    '1000130290': 'Al-29',                 # Aluminum-29
    '1000130299': 'Al-29m',                # Aluminum-29 (Metastable)
    '1000130300': 'Al-30',                 # Aluminum-30
    '1000130309': 'Al-30m',                # Aluminum-30 (Metastable)
    '1000130310': 'Al-31m',                # Aluminum-31 (Metastable)
    '1000130320': 'Al-32',                 # Aluminum-32
    '1000130329': 'Al-32m',                # Aluminum-32 (Metastable)
    '1000130330': 'Al-33',                 # Aluminum-33
    '1000130340': 'Al-34',                 # Aluminum-34
    '1000130350': 'Al-35',                 # Aluminum-35
    '1000140260': 'Si-26',                 # Silicon-26
    '1000140270': 'Si-27',                 # Silicon-37
    '1000140280': 'Si-28',                 # Silicon-28
    '1000140290': 'Si-29',                 # Silicon-29
    '1000140300': 'Si-30',                 # Silicon-30
    '1000140310': 'Si-31',                 # Silicon-31
    '1000140320': 'Si-32',                 # Silicon-32
    '1000140329': 'Si-32m',                # Silicon-32 (Metastable)
    '1000140330': 'Si-33',                 # Silicon-33
    '1000140339': 'Si-33m',                # Silicon-33 (Metastable)
    '1000140340': 'Si-34',                 # Silicon-34
    '1000140349': 'Si-34m',                # Silicon-34 (Metastable)
    '1000140350': 'Si-35',                 # Silicon-35
    '1000140360': 'Si-36',                 # Silicon-36
    '1000140370': 'Si-37',                 # Silicon-37
    '1000150280': 'P-28',                  # Phosphorus-28
    '1000150290': 'P-29',                  # Phosphorus-29
    '1000150300': 'P-30',                  # Phosphorus-30
    '1000150310': 'P-31',                  # Phosphorus-31
    '1000150320': 'P-32',                  # Phosphorus-32
    '1000150330': 'P-33',                  # Phosphorus-33
    '1000150340': 'P-34',                  # Phosphorus-34
    '1000150349': 'P-34m',                 # Phosphorus-34 (Metastable)
    '1000150350': 'P-35',                  # Phosphorus-35
    '1000150360': 'P-36',                  # Phosphorus-36
    '1000150370': 'P-37',                  # Phosphorus-37
    '1000150380': 'P-38',                  # Phosphorus-38
    '1000160290': 'S-29',                  # Sulfur-29
    '1000160300': 'S-30',                  # Sulfur-30
    '1000160310': 'S-31',                  # Sulfur-31
    '1000160320': 'S-32',                  # Sulfur-32
    '1000160330': 'S-33',                  # Sulfur-33
    '1000160340': 'S-34',                  # Sulfur-34
    '1000160350': 'S-35',                  # Sulfur-35
    '1000160359': 'S-35m',                 # Sulfur-35 (Metastable)
    '1000160360': 'S-36',                  # Sulfur-36
    '1000160369': 'S-36m',                 # Sulfur-36 (Metastable)
    '1000160370': 'S-37',                  # Sulfur-37
    '1000160380': 'S-38',                  # Sulfur-38
    '1000160390': 'S-39',                  # Sulfur-39
    '1000160400': 'S-40',                  # Sulfur-40
    '1000170320': 'Cl-32',                 # Chlorine-32
    '1000170330': 'Cl-33',                 # Chlorine-33
    '1000170340': 'Cl-34',                 # Chlorine-34
    '1000170341': 'Cl-34m',                # Chlorine-34 (Metastable)
    '1000170349': 'Cl-34m',                # Chlorine-34 (Metastable)
    '1000170360': 'Cl-36',                 # Chlorine-36
    '1000170369': 'Cl-36m',                # Chlorine-36 (Metastable)
    '1000170409': 'Cl-40',                 # Chlorine-40
    '1000170350': 'Cl-35',                 # Chlorine-35
    '1000170370': 'Cl-37',                 # Chlorine-37
    '1000170380': 'Cl-38',                 # Chlorine-38
    '1000170381': 'Cl-38m',                # Chlorine-38 (Metastable)
    '1000170389': 'Cl-38',                 # Chlorine-38
    '1000170390': 'Cl-39',                 # Chlorine-39
    '1000170400': 'Cl-40',                 # Chlorine-40
    '1000180330': 'Ar-33',                 # Argon-33
    '1000180340': 'Ar-34',                 # Argon-34
    '1000180350': 'Ar-35',                 # Argon-35
    '1000180360': 'Ar-36',                 # Argon-36
    '1000180370': 'Ar-37',                 # Argon-37
    '1000180379': 'Ar-37m',                # Argon-37 (Metastable)
    '1000180380': 'Ar-38',                 # Argon-38
    '1000180390': 'Ar-39',                 # Argon-39
    '1000180399': 'Ar-39m',                # Argon-39 (Metastable)
    '1000180400': 'Ar-40',                 # Argon-40
    '1000180409': 'Ar-40m',                # Argon-40 (Metastable)
    '1000180410': 'Ar-41',                 # Argon-41
    '1000180420': 'Ar-42',                 # Argon-42
    '1000190350': 'K-35',                  # Potassium-35
    '1000190360': 'K-36',                  # Potassium-36
    '1000190370': 'K-37',                  # Potassium-37
    '1000190380': 'K-38',                  # Potassium-38
    '1000190382': 'K-38m',                 # Potassium-38 (Metastable)
    '1000190389': 'K-38m',                 # Potassium-38 (Metastable)
    '1000190390': 'K-39',                  # Potassium-39
    '1000190400': 'K-40',                  # Potassium-40
    '1000190409': 'K-40m',                 # Potassium-40 (Metastable)
    '1000190410': 'K-41',                  # Potassium-41
    '1000190419': 'K-41m',                 # Potassium-41 (Metastable)
    '1000190420': 'K-42',                  # Potassium-42
    '1000190429': 'K-42m',                 # Potassium-42 (Metastable)
    '1000190430': 'K-43',                  # Potassium-43
    '1000190439': 'K-43m',                 # Potassium-43 (Metastable)
    '1000200390': 'Ca-39',                 # Calcium-39
    '1000200400': 'Ca-40',                 # Calcium-40
    '1000200410': 'Ca-41',                 # Calcium-41
    '1000200420': 'Ca-42',                 # Calcium-42
    '1000200429': 'Ca-42',                 # Calcium-42
    '1000200430': 'Ca-43',                 # Calcium-43
    '1000200440': 'Ca-44',                  # Calcium-44

    # Update From Here
    '1000120340': 'Mg-34',
    '1000010060': 'H-6',
    '1000070110': 'N-11',
    '1000100199': 'Ne-19',
    '1000110269': 'Na-26',
    '1000150270': 'P-27',
    '1000130240': 'Al-24',
    '1000210450': 'Sc-45'                                   }


                                                        
