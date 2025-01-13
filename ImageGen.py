
#This script should create the images, aranged in subfolders where the kind of particles are set. 
#The main concern here is first to create two datasets: 
# - one for the current discrimination (Neutral CUrrent and CHarged CUrrent)
# - another one for type discrimination (reaction number)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
import os
import h5py 
import cv2

def imcrop(bool,reaction,spath,imgstr,size):
    
    dict= {
        1:"QES",
        2:"1Kaon",
        3:"DIS",
        4: "RES",
        5: "COH",
        6: "DFR",
        7: "NuEEL",
        8: "IMD",
        9:"AmNuGamma",
        10:"MEC",
        11:"CEvNS",
        12: "IBD",
        13: "GLR",
        14: "IMDAnh",
        15: "PhotonCOH",
        16: "PhotonRES"
    }
    a,b,c,d =size
    image=cv2.imread(spath)

    if bool==True:
        cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+'/ImagesForCurrent/CC/'+imgstr,image[a:b,c:d])
    elif bool==False: 
        cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+'/ImagesForCurrent/NC/'+imgstr,image[a:b,c:d])
    #Now we sort them out
    image=cv2.imread(spath)
    cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+'/ImagesForReaction/'+str(dict[int(reaction)])+'/'+imgstr,image[a:b,c:d])


if __name__== "__main__":
    #Now we can start the main program
#We create the directories were the images are gonna be saved.
    if os.path.exists("CleanImages")==False:
        os.mkdir("CleanImages")
#First the clean images, were the tag and all of them will be placed.
#Then the images for current discrimination
    if os.path.exists("ImagesForCurrent")==False:
        os.mkdir("ImagesForCurrent")
        os.mkdir("ImagesForCurrent/CC")
        os.mkdir("ImagesForCurrent/NC")
#Finally the kinds that are supported by GENIE

    if os.path.exists("ImagesForReaction")==False:
        os.mkdir("ImagesForReaction")
        os.mkdir("ImagesForReaction/QES") #1
        os.mkdir("ImagesForReaction/1Kaon") #2
        os.mkdir("ImagesForReaction/DIS") #3
        os.mkdir("ImagesForReaction/RES") #4
        os.mkdir("ImagesForReaction/COH") #5
        os.mkdir("ImagesForReaction/DFR") #6
        os.mkdir("ImagesForReaction/NuEEL") #7
        os.mkdir("ImagesForReaction/IMD") #8
        os.mkdir("ImagesForReaction/AmNuGamma") #9
        os.mkdir("ImagesForReaction/MEC") #10
        os.mkdir("ImagesForReaction/CEvNS") #11
        os.mkdir("ImagesForReaction/IBD") #12
        os.mkdir("ImagesForReaction/GLR") #13
        os.mkdir("ImagesForReaction/IMDAnh") #14
        os.mkdir("ImagesForReaction/PhotonCOH") #15
        os.mkdir("ImagesForReaction/PhotonRes") #16

        
#The other kind associated with reactions will be created after.

    my_path = os.path.dirname(os.path.abspath(__file__))
#Now let's import the file
    sim_file='/home/enrique/Documents/WorkStuff/Work1/MicroProdN1p2_NDLAr_1E18_RHC.convert2h5.nu.0000100.EDEPSIM.hdf5'
    sim_h5= h5py.File(sim_file,'r')
#Split the contents of each.
    mc_hdr = sim_h5['mc_hdr']
    mc_stack = sim_h5['mc_stack']
    segments = sim_h5['segments']
    trajectories = sim_h5['trajectories']
    vertices= sim_h5['vertices']
#Now we want to get the unique id's of the events inside the events. 
    events_ids, counts=np.unique(segments['event_id'],return_counts=True)
#This splits into the kind of reaction
    kind=mc_hdr.dtype.names[8:14]
    print("The number of events is: " , len(events_ids))
    # We normalize the color scale of all the images being generated in a batch.
    norm=colors.Normalize(0,np.max(segments['dE']))
    # With this info we can start generating the images.
    # We scan each of the event ids inside the image
    j=0
    for evid in events_ids:
        seg_event=segments[segments['event_id']==evid]
        info_event=mc_hdr[mc_hdr['event_id']==evid]
        #From info_event we get the info of the vertex.
        vertex_ids, counts=np.unique(seg_event['vertex_id'],return_counts=True)
        
        for vid in vertex_ids:
            int_vert=seg_event[seg_event['vertex_id']==vid]
            filt_vert=int_vert[int_vert['dE']>1]
            info_vertex=info_event[info_event['vertex_id']==vid]
            #And here we get the int associated with the reaction
            reaction =int(info_vertex['reaction'])
            name=str(reaction)
            for kin in kind:
                if info_vertex['isCC']==True:
                    name=name+str(kin)
            print(evid,vid)
            figure, bars= plt.subplots()
            bars.set_title("Interaction kind: "+ str(name))
            bars.set_xlabel(" Z (cm)" )
            bars.set_ylabel(" Y (cm)")
            bars.set_xlim(420,950)
            bars.set_ylim(-220,100)
            bars.set_facecolor("black")
        #plt.rcParams['figure.figsize'] = [10, 10]
            bars.scatter(filt_vert['z'],filt_vert['y'],c=filt_vert['dE'],cmap='grey',s=1)
            figure.colorbar(cm.ScalarMappable(norm=norm,cmap='grey'),ax=bars,label="Energy Deposition (MeV)")

            figstr=str(name)+str(j)+".png"
            figure.savefig(my_path+'/CleanImages/'+figstr,dpi=100)
            spath=my_path+'/CleanImages/'+figstr
            #Split into CC or NC
            if info_vertex['isCC']==True:
                bool=True
            else: 
                bool=False

            imcrop(bool,reaction,spath,figstr,(58,428,78,478))
            j+=1

#Now what will follow is to have the images in all of the datasets.
