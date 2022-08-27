import ROOT
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import nflows
import time
import os
import awkward as ak
import mynflow
import many_upsampling
import pathlib

if __name__=='__main__':

    root = '/gpfs/ddn/srm/cms//store/mc/RunIIAutumn18NanoAODv6/DY2JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/'
	#'/gpfs/ddn/cms/user/fvaselli/VBFHToMuMu_powheg/'
    new_root = '/gpfs/ddn/cms/user/fvaselli/DY2JetsToLL_upsampled/'
    ttbar_training_files = ['250000/047F4368-97D4-1A4E-B896-23C6C72DD2BE.root', '240000/B38E7351-C9E4-5642-90A2-F075E2411B00.root',
                            '230000/DA422D8F-6198-EE47-8B00-1D94D97950B6.root', '230000/393066F3-390A-EC4A-9873-BF4D4D7FBE4F.root',
                            '230000/12C9A5BF-1608-DA48-82E9-36F18051CE31.root', '230000/12C8AFA5-B554-9540-8603-2DF948304880.root',
                            '250000/02B1F58F-7798-FB44-BF80-56C3DC1B6E52.root', '230000/78137863-DAD0-E740-B357-D88AF92BE59F.root',
                            '230000/91456D0B-2FDE-2B4F-8C7A-8E60260480CD.root']
    files_paths =  [os.path.join(d, f) for d in os.listdir(root) for f in os.listdir(os.path.join(root, d))] # [f for f in os.listdir(root)]
    files_paths = files_paths[10:]
# [os.path.join(d, f) for d in os.listdir(root) for f in os.listdir(os.path.join(root, d))] # = [x for x in pathlib.Path(root).glob('**/*')]

    # optionally remove training files if we are generating ttbar dataset
    # files_paths = [path for path in files_paths if path not in ttbar_training_files]
    # take remaining files
    #files_paths = ['HMM7.root']
    print(files_paths)
    files_paths = files_paths[:]
    print(files_paths)
    print(f'We will process a total of {len(files_paths)} files')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	

    jet_flow, _, _, _, trh, tsh = mynflow.load_model("../nflows/saves", "model_jets_final_@epoch_420.pt")
    muon_flow, _, _, _, trh, tsh = mynflow.load_model("../nflows/saves", "model_muons_final_@epoch_580.pt")

    for path in tqdm(files_paths):
        path_str = str(path) # shouldn't be needed
        many_upsampling.nbd(jet_flow, muon_flow, root, path_str, new_root)
