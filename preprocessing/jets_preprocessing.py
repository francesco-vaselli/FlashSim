import uproot
import pandas as pd
import h5py 
import numpy as np
import sys


if __name__=='__main__':
    f = sys.argv[1]  
    tree = uproot.open(f"MJetsA{f}.root:MJets", num_workers=20)
    vars_to_save = tree.keys()
    print(vars_to_save)

    # define pandas df for fast manipulation 
    df = tree.arrays(library="pd").reset_index(drop=True).astype('float32').dropna()
    print(df)
    
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]	
    # apply saturations
    a = df['MJet_massRatio'].values
    df['MJet_massRatio'] = np.where(a > 7, 7, a)
    a = df['MJet_massRatio'].values
    df['MJet_massRatio'] = np.where(a < 0, 0, a)
    a = df['MJet_ptRatio'].values
    df['MJet_ptRatio'] = np.where(a > 3, 3, a)
    a = df['MJet_ptRatio'].values
    df['MJet_ptRatio'] = np.where(a < 0, 0, a)
    
    maskB = (df["MJet_btagDeepB"]<0) 
    maskC = (df["MJet_btagDeepB"]<0) 
    maskCSV = (df["MJet_btagDeepB"]<0) 
    maskQGL = (df["MJet_qgl"]<0) 
    df.loc[maskB, "MJet_btagDeepB"] = -0.1
    df.loc[maskC, "MJet_btagCSVV2"] = -0.1
    df.loc[maskCSV, "MJet_btagDeepC"] = -0.1
    df.loc[maskQGL, "MJet_qgl"] = -0.1

    # apply transformations
    df['MJet_jetId'] = df['MJet_jetId'].apply(lambda x: x + 0.1*np.random.normal())
    df['MJet_puId'] = df['MJet_puId'].apply(lambda x: x + 0.1*np.random.normal())
    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]	 
    print(df)
        
    # open hdf5 file for saving
    f = h5py.File(f'Ajets_and_muons{f}+.hdf5','w')

    dset = f.create_dataset("data", data=df.values)#, dtype='f4')

    f.close()
