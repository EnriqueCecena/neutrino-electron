# Event Discrimination and CNN dataset construction script.

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import numpy as np
from matplotlib import cm, colors
import cv2
import os
import glob
import sys

#After getting everything imported, we declare the main program.

#This function crops the generated images to just the contents of the plots (without axes, etc.)

def imcrop(bool,spath,imgstr,size):
    a,b,c,d =size
    image=cv2.imread(spath)
    if bool==True:
        cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+'/ImagesForCNN/CC/'+imgstr,image[a:b,c:d])
    else: 
        cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+'/ImagesForCNN/NC/'+imgstr,image[a:b,c:d])


if __name__== "__main__":
# We are gonna write the code for one file, a for loop should be enought to loop it through all the other files and
# generate the other images.
    if os.path.exists("ImagesVertex")==False:
        os.mkdir("ImagesVertex")
#Let's split all of the events in two categories CCevent and NCevent and place each in a different folder 
    if os.path.exists("ImagesForCNN")==False:       
        os.mkdir("ImagesForCNN")
        os.mkdir("ImagesForCNN/NC")
        os.mkdir("ImagesForCNN/CC")

    my_path = os.path.dirname(os.path.abspath(__file__))
# We read the file
    sim_file='/home/enrique/Documents/WorkStuff/Work1/MicroProdN1p2_NDLAr_1E18_RHC.convert2h5.nu.0000100.EDEPSIM.hdf5'
    sim_h5= h5py.File(sim_file,'r')
# We do not need to inspect the contents of the file as they're already known.
# We declare the datasets. 
    mc_hdr = sim_h5['mc_hdr']
    mc_stack = sim_h5['mc_stack']
    segments = sim_h5['segments']
    trajectories = sim_h5['trajectories']
    vertices= sim_h5['vertices']
#The main ones that we need are mc_hdr and segments. 

    events_ids, counts=np.unique(segments['event_id'],return_counts=True)
    kind=mc_hdr.dtype.names[8:14]
    print("The number of events is: " , len(events_ids))
# We normalize the color scale of all the images being generated in a batch.
    norm=colors.Normalize(0,np.max(segments['dE']))
# With this info we can start generating the images.
# We scan each of the event ids inside the image

#Counter for the name of the file.
    j=0
    for evid in events_ids:
        seg_event=segments[segments['event_id']==evid]
        info_event=mc_hdr[mc_hdr['event_id']==evid]
        vertex_ids, counts=np.unique(seg_event['vertex_id'],return_counts=True)
        
        for vid in vertex_ids:
            int_vert=seg_event[seg_event['vertex_id']==vid]
            filt_vert=int_vert[int_vert['dE']>1]
            info_vertex=info_event[info_event['vertex_id']==vid]
            name=[]
            for kin in kind:
                if info_vertex[kin]==True:
                    name.append(kin)
            print(evid,vid)
            figure, bars= plt.subplots()
            bars.set_title("Interaction kind: "+ ' '.join(name))
            bars.set_xlabel(" Z (cm)" )
            bars.set_ylabel(" Y (cm)")
            bars.set_xlim(420,950)
            bars.set_ylim(-220,100)
    #plt.rcParams['figure.figsize'] = [10, 10]
            bars.scatter(filt_vert['z'],filt_vert['y'],c=filt_vert['dE'],cmap='grey',s=1)
            figure.colorbar(cm.ScalarMappable(norm=norm,cmap='grey'),ax=bars,label="Energy Deposition (MeV)")
            figstr=''.join(name)+str(j)+".png"
            figure.savefig(my_path+'/ImagesVertex/'+figstr,dpi=200)
            spath=my_path+'/ImagesVertex/'+figstr
            if len(name)>0:
                if name[0]=="isCC":
                    bool=True
                else: 
                    bool=False
            else:
                continue    
            imcrop(bool,spath,figstr,(115,850,160,950))
            j+=1


#The next thing that this program should do is to create the hdf5 dataset using the images inside of the folder.
#We declare the name of the .h5 file that's gonna contain the info.

    h5_file="Currents.h5"
    #The files to be added into the dataset are from two folders
    nfiles1=len(glob.glob(my_path+'/ImagesForCNN/CC/*.png'))
    nfiles2=len(glob.glob(my_path+'/ImagesForCNN/NC/*.png'))
    n=nfiles1+nfiles2
    with h5py.File(h5_file,'w') as h5f:
        img_ds=h5f.create_dataset('images',shape=(n,2,735,790))
        print(img_ds)
        for cnt, ifile in enumerate(glob.iglob(my_path+'/ImagesForCNN/CC/*.png')):
            print(cnt,ifile)
            image_resized = cv2.imread(ifile,cv2.IMREAD_GRAYSCALE)
            img_ds[cnt,:,:]=image_resized
            cnt=cnt
        for cnt2, ifile in enumerate(glob.iglob(my_path+'/ImagesForCNN/NC/*.png')):
            print(cnt+cnt2,ifile)
            image_resized = cv2.imread(ifile,cv2.IMREAD_GRAYSCALE)
            img_ds[cnt,:,:]=image_resized
        h5f.close()
    



