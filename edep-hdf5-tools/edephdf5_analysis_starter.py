#######################################################################
##
## edephdf5_analysis_starter.py. Alexander Booth, QMUL, Jan.2025.
##
## A script to work with .EDEPSIM.hdf5 files from the
## DUNE ND production chain for NDLAr. Details of what
## is currently available in these files can be found at
## the following link:
##
## https://github.com/DUNE/2x2_sim/wiki/File-data-definitions#mc-truth
##
#######################################################################


from matplotlib.backends.backend_pdf import PdfPages

import argparse
import h5py
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np


# Set axis label size globally.
matplotlib.rcParams['axes.labelsize'] = 20


# Construct a string with all of the particles
# involved in a particular interaction. There
# is almost always an argon so the default is
# to exclude argon PDG from the string.
def build_process_string(pdgs, statuses, exclude_argon=True):
    primaries   = np.where(statuses==0)   
    secondaries = np.where(statuses==1)   

    # Count the protons and neutrons (can be loads of them, the string would 
    # sometimes be very long if we didn't do this).
    primary_protons = np.where(pdgs[primaries]==2212)
    n_proton_primaries = np.size(primary_protons)
    secondary_protons = np.where(pdgs[secondaries]==2212)
    n_proton_secondaries = np.size(secondary_protons)
    primary_neutrons = np.where(pdgs[primaries]==2112)
    n_neutron_primaries = np.size(primary_neutrons)
    secondary_neutrons = np.where(pdgs[secondaries]==2112)
    n_neutron_secondaries = np.size(secondary_neutrons)

    ret = ""
    if n_neutron_primaries: ret += f"(2112 x {n_neutron_primaries}) + "
    if n_proton_primaries: ret += f"(2212 x {n_proton_primaries}) + "
    for pdg in pdgs[primaries]:
        if exclude_argon and pdg == 1000180400: continue
        if pdg == 2112 or pdg == 2212: continue
        ret += f'{pdg} + '
    ret = ret[:-2]
    ret += " -> "
    if n_neutron_secondaries: ret += f"(2112 x {n_neutron_secondaries}) + "
    if n_proton_secondaries: ret += f"(2212 x {n_proton_secondaries}) + "
    for pdg in pdgs[secondaries]:
        if exclude_argon and pdg == 1000180400: continue
        if pdg == 2112 or pdg == 2212: continue
        ret += f'{pdg} + '
    ret = ret[:-2]

    return ret


# Based on the minimum and maximum positions of energy depositions in
# an interaction and the detector resolution, calculate the size of
# the pixel map to draw.
def get_pixel_map_dimensions(horizontal_axis, vertical_axis):
    # Roughly the pixel separation in cm.
    detector_resolution = 0.5

    horizontal_min = horizontal_axis[np.argmin(horizontal_axis)]
    horizontal_max = horizontal_axis[np.argmax(horizontal_axis)]
    vertical_min = vertical_axis[np.argmin(vertical_axis)]
    vertical_max = vertical_axis[np.argmax(vertical_axis)]

    real_horizontal_size = horizontal_max-horizontal_min
    n_bins_horizontal    = math.ceil(real_horizontal_size/detector_resolution)
    real_vertical_size   = vertical_max-vertical_min
    n_bins_vertical      = math.ceil(real_vertical_size/detector_resolution)

    return [n_bins_horizontal, n_bins_vertical], [[horizontal_min, horizontal_max],[vertical_min, vertical_max]] 


# Get the rough edges of the active LAr region in NDLAr.
def get_rough_active_region(dimension):
    if dimension=="x": return [-360.0,360.0]
    if dimension=="y": return [-220.0, 84.0]
    if dimension=="z": return [ 410.0,920.0]
    raise "dimension must be x, y or z"


# Parse the command line arguments.
parser = argparse.ArgumentParser()

parser.add_argument("-e", "--evds_only", action='store_true', help="Only make event display style plots.")
parser.add_argument("-f", "--input_hdf5", type=str, required=True, help="Path and file name of .EDEPSIM.hdf5 file of interest.")
parser.add_argument("-i", "--interaction_limit", type=int, default=10, help="Only consider the first interaction_limit interactions per spill for event display making. Set to -1 to consider all.")
parser.add_argument("-o", "--outfile_stub", type=str, help="Add a string to the end of the output pdf file name - use for unique output naming.")
parser.add_argument("-s", "--spill_limit", type=int, default=10, help="Only consider the first spill_limit spills for event display making. Set to -1 to consider all.")

args = parser.parse_args()

# Open the input hdf5 as read only. 
f_in_path = args.input_hdf5
f_in = h5py.File(f_in_path, 'r')


# Set up output pdf.
outfile_stub = ""
if args.outfile_stub: outfile_stub = "_" + args.outfile_stub
with PdfPages(f"edephdf5_analysis_starter{outfile_stub}.pdf") as f_out:
    

    # Get mc_hdr dataset.
    mc_hdr = f_in["mc_hdr"]
    # Get mc_stack dataset.
    mc_stack = f_in["mc_stack"]
    # Get segments dataset.
    segments = f_in["segments"]
    
    
    if not args.evds_only:
        # 1D histograms from mc_hdr dataset.
        plt.hist(mc_hdr["reaction"], bins=20, range=[0, 20])
        plt.xlabel("Interaction Type")
        plt.ylabel("Interactions")
        f_out.savefig()
        plt.close()
        
        plt.hist(mc_hdr["nu_pdg"], bins=40, range=[-20, 20])
        plt.xlabel("Primary Neutrino PDG")
        plt.ylabel("Interactions")
        f_out.savefig()
        plt.close()
        
        plt.hist(mc_hdr["Enu"]/1000.0, bins=100)
        plt.xlabel("Primary Neutrino Energy (GeV)")
        plt.ylabel("Interactions")
        f_out.savefig()
        plt.close()
        
 
        # 1D histograms from segments dataset.
        plt.hist(segments["dEdx"], bins=1000)
        plt.xlabel("dE/dx (MeV / cm)")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
 
        plt.hist(segments["dEdx"], bins=1000, range=[0, 5.0])
        plt.xlabel("dE/dx (MeV / cm)")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
 
        plt.hist(segments["dEdx"], bins=1000, range=[0, 0.1])
        plt.xlabel("dE/dx (MeV / cm)")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
 
        plt.hist(segments["dE"], bins=1000)
        plt.xlabel("dE (MeV)")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
 
        plt.hist(segments["dE"], bins=1000, range=[0, 1.5])
        plt.xlabel("dE (MeV)")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
 
        plt.hist(segments["dE"], bins=1000, range=[0, 0.1])
        plt.xlabel("dE (MeV)")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
 
        plt.hist(segments["dx"], bins=1000)
        plt.xlabel("dx (cm)")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
 
        plt.hist(segments["dx"], bins=1000, range=[0, 0.1])
        plt.xlabel("dx (cm)")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
 
        plt.hist(segments["pdg_id"], bins=1000)
        plt.xlabel("PDG of Particle Leaving Deposit")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
 
        plt.hist(segments["pdg_id"], bins=50, range=[-25,25])
        plt.xlabel("PDG of Particle Leaving Deposit")
        plt.ylabel("Segments")
        plt.yscale('log')
        f_out.savefig()
        plt.close()
    
    
    # Get a list of all of the spills in the file, indexed by
    # event_id.
    event_ids = np.unique(segments['event_id'])
    event_count = 0
    # Loop over spills.
    for event_id in event_ids:
        if args.spill_limit!=-1 and event_count > args.spill_limit-1: break
        print(f"Working on spill {event_count}...")

        # Get a list of all of the energy deposits, segments, in
        # the current spill.
        event_mask     = segments['event_id']==event_id 
        event_segments = segments[event_mask]

        # Get a list of all neutrino interactions in the current
        # spill, indexed by vertex_id.
        vertex_ids = np.unique(segments["vertex_id"][event_mask])
        n_vertex_ids = np.size(vertex_ids)

        x = np.array(event_segments['x'])
        y = np.array(event_segments['y'])
        z = np.array(event_segments['z'])
        c = np.array(event_segments['dEdx'])

        fig = plt.figure(figsize=(20,20))
        ax  = fig.add_subplot(projection='3d')
        ax.scatter(z, x, y, c=c, vmax=5.0, cmap='copper', s=1, marker=".")
        ax.set_title(f"Spill: {event_id}. {n_vertex_ids} Neutrino Interactions.", fontsize=20)
        ax.set_xlabel('z [cm]')
        ax.set_ylabel('x [cm]')
        ax.set_zlabel('y [cm]')
        ax.set_ylim(get_rough_active_region("x"))
        ax.set_zlim(get_rough_active_region("y"))
        ax.set_xlim(get_rough_active_region("z"))
        f_out.savefig()

        # Make the z data axis run exactly horizontal.
        ax.elev = 0
        # Make the x data axis run directly into the figure.
        ax.azim = 270
        ax.set_xlabel("")
        f_out.savefig()
        plt.close()

        
        # Now loop over neutrino interactions in the current spill.
        vertex_count = 0
        for vertex_id in vertex_ids:
            if args.interaction_limit!=-1 and vertex_count > args.interaction_limit-1: break
            vertex_id_mc_hdr_mask   = (mc_hdr["vertex_id"]==vertex_id) & (mc_hdr["event_id"]==event_id)
            vertex_id_mc_stack_mask = (mc_stack["vertex_id"]==vertex_id) & (mc_stack["event_id"]==event_id)
            vertex_id_segments_mask = event_segments["vertex_id"]==vertex_id

            interaction_mc_hdr   = mc_hdr[vertex_id_mc_hdr_mask]
            interaction_mc_stack = mc_stack[vertex_id_mc_stack_mask]
            interaction_segments = event_segments[vertex_id_segments_mask]

            Enu         = round(interaction_mc_hdr["Enu"][0]/1000.,1)
            reaction    = interaction_mc_hdr["reaction"][0]
            part_pdg    = interaction_mc_stack["part_pdg"]
            part_status = interaction_mc_stack["part_status"]


            x = interaction_segments['x']
            z = interaction_segments['z']
            c = interaction_segments['dEdx']
            bins_xz, range_xz = get_pixel_map_dimensions(x, z)

            y = interaction_segments['y']
            bins_yz, range_yz = get_pixel_map_dimensions(y, z)

            process = build_process_string(part_pdg, part_status)

            fig, ax = plt.subplots(1, 2, figsize=(20,20))
            fig.suptitle(f"Neutrino Energy: {Enu} GeV. Interaction: {reaction}. Process: {process}", fontsize=20)
            # Draw x vs. z.
            plt.subplot(1, 2, 1)
            plt.hist2d(x, z, weights=c, bins=bins_xz, range=range_xz, cmap='Greys', vmax=5.0)
            plt.xlabel("x (cm)")
            plt.ylabel("z (cm)")
            # Draw y vs. z.
            plt.subplot(1, 2, 2)
            plt.hist2d(y, z, weights=c, bins=bins_yz, range=range_yz, cmap='Greys', vmax=5.0)
            plt.xlabel("y (cm)")
            plt.ylabel("z (cm)")
            plt.colorbar(label="dE/dx (MeV/cm)")

            f_out.savefig()
            plt.close()


            vertex_count += 1


        event_count += 1
