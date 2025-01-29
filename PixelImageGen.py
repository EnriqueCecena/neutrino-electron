#
#
#In this code we shall generate the code to create images for training.
#Using the enriched files

#Import the libraries.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
import os
import h5py 
import random
import string



#This function generates the folders into which the images are gonna be saved.
def folder_generate(my_path,react_dict):
    react_array=react_dict.items()
    react_array=list(react_array)
    react_array=np.array(react_array)
    os.makedirs(my_path+'/TrainImages/',exist_ok=True)
#Now inside this main folder we generate each of the folders that are gonna contain the images. 
#first the neutral current.
    os.makedirs(my_path+'/TrainImages/NC',exist_ok=True)
#and to this quickly.
    for i in range(0, len(react_array)):
        os.makedirs(my_path+'/TrainImages/'+str(react_array[i][1]),exist_ok=True)

def work_file(my_path,react_dict,sim_file):
    #We read the file
    sim_h5= h5py.File(sim_file,'r')
    print('\n----------------- File content -----------------')
    print('File:',sim_file)
    print('Keys in file:',list(sim_h5.keys()))
    for key in sim_h5.keys():
        print('Number of',key,'entries in file:', len(sim_h5[key]))
    print('------------------------------------------------\n')

    #Split the contents of the file
    mc_hdr = sim_h5['mc_hdr']
    mc_stack = sim_h5['mc_stack']
    segments = sim_h5['segments']
    trajectories = sim_h5['trajectories']
    vertices= sim_h5['vertices']
    #Now we want to get the unique id's of the events inside the events. 
    events_ids, counts=np.unique(segments['event_id'],return_counts=True)
    print("Kinds of objects inside the mc_hdr dataset: ")
    print("")
    print(mc_hdr.dtype)
    print("")
    print("number of events")
    print(events_ids,counts)
    print("The number of events is: " , len(events_ids))
    # We normalize the color scale of all the images being generated in a batch.
    norm=colors.Normalize(0,np.max(segments['dE']))
    kind=mc_hdr.dtype.names[8:14]
    #We predefine some quantities for the pixeled image
    z_range=[415,918] #measurements are in cm
    y_range=[-216,83]
    #We generate the sizes of the bins, as we've discussed in other chats the size should be equivalent to a 1.5cm box 
    # according to the pixel size and the detector size. 
    px=1/plt.rcParams['figure.dpi']
    x_bins=np.arange(z_range[0],z_range[1],1.5)
    y_bins=np.arange(y_range[0],y_range[1],1.5)
    #The final size would be an image of 336x200 pixels

    #Now per each of the events inside the file, we use:
    for id in events_ids:
    #We grab the info of the event from the mc header
        info_event=mc_hdr[mc_hdr['event_id']==id]
    #asign the reaction and the charge
        reaction=info_event['reaction']
        boolean=info_event['isCC']
        event=segments[segments['event_id']==id]
        filt_event=event[event['dE']>1]
    #Now that we've filtered the info we must generate the images.
        grid, xedges,yedges= np.histogram2d(filt_event['z'],filt_event['y'],bins=[x_bins,y_bins],weights=filt_event['dE'])
        fig, ax = plt.subplots(figsize=(336*px,200*px))
        ax.imshow(grid.T, origin='lower', extent=(z_range[0], z_range[1], y_range[0], y_range[1]),cmap='grey', aspect='auto', interpolation='nearest')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #Now we must save the images on the right folders and add a unique name for it
        #to give a unique name
        rand=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        if int(reaction) in react_dict:
            if boolean==True:
                figpath=my_path+'/TrainImages/'+str(react_dict[int(reaction)])+'/'+str(reaction)+str(rand)+'.png'
            else:
                figpath=my_path+'/TrainImages/NC'+'/'+str(reaction)+str(rand)+'.png'
            fig.savefig(figpath)
            print(id)
            plt.close('all')
        else:
            continue





#We create the main program
if __name__== "__main__":
#We grab the path in which we are.
    my_path =os.path.dirname(os.path.abspath(__file__))
#We add the dictionary that associates the reaction types to the corresponding labels.
    react_dict={
        1:"QES",
        3:"DIS",
        4: "RES",
        5: "COH",
        7: "NuEEL",
        8: "IMD",
        10:"MEC"
    }
# For the main purpose of generating the folders, we create a copy of this dictionary
# into an array.

#We generate the folders
    folder_generate(my_path,react_dict)
#Now we need to import the file and send it to process
    sim_file='/home/enrique/Documents/neutrino-electron/data2/MicroProdN3p2_NDLAr_2E22_FHC_NuE.convert2h5.nu.0000004.EDEPSIM.hdf5'
    work_file(my_path,react_dict,sim_file)