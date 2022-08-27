import uproot
import pandas as pd
import h5py 
import numpy as np
import sys


if __name__=='__main__':
	f = sys.argv[1] 
	tree = uproot.open(f"MMuonsA{f}.root:MMuons", num_workers=20)
	vars_to_save = tree.keys()
	print(vars_to_save)

	# define pandas df for fast manipulation 
	df = tree.arrays(library="pd").reset_index(drop=True).astype('float32').dropna()
	print(df)
	
	# we do not drop any flag
	# df = df.drop(["MGenPart_statusFlags2", "MGenPart_statusFlags12", "MGenPart_statusFlags14"], axis=1)
	df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]	
	# apply saturations 
	a = df["MMuon_dxyErr"].values 
	df["MMuon_dxyErr"] = np.where(a > 1, 1, a)
	a = df["MMuon_dz"].values 
	df["MMuon_dz"] = np.where(a > 20, 20, a)
	a = df["MMuon_dz"].values 
	df["MMuon_dz"] = np.where(a < -20, -20, a)
	a = df["MMuon_dzErr"].values 
	df["MMuon_dzErr"] = np.where(a > 1, 1, a)
	a = df["MMuon_ip3d"].values 
	df["MMuon_ip3d"] = np.where(a > 1, 1, a)
	a = df["MMuon_jetPtRelv2"].values 
	df["MMuon_jetPtRelv2"] = np.where(a > 200, 200, a)
	a = df["MMuon_jetRelIso"].values 
	df["MMuon_jetRelIso"] = np.where(a > 100, 100, a)
	a = df["MMuon_pfRelIso03_all"].values 
	df["MMuon_pfRelIso03_all"] = np.where(a > 100, 100, a)
	a = df["MMuon_pfRelIso03_chg"].values 
	df["MMuon_pfRelIso03_chg"] = np.where(a > 40, 40, a)
	a = df["MMuon_pfRelIso04_all"].values 
	df["MMuon_pfRelIso04_all"] = np.where(a > 70, 70, a)
	a = df["MMuon_ptErr"].values 
	df["MMuon_ptErr"] = np.where(a > 300, 300, a)
	a = df["MMuon_sip3d"].values 
	df["MMuon_sip3d"] = np.where(a > 1000, 1000, a)

	#  what if we trained without transforms?
		
	df["MGenMuon_pt"] = df["MGenMuon_pt"].apply(lambda x: np.log(x))
	df["MClosestJet_pt"] = df["MClosestJet_pt"].apply(lambda x: np.log1p(x))
	df["MClosestJet_mass"] = df["MClosestJet_mass"].apply(lambda x: np.log1p(x))
	df["Pileup_sumEOOT"] = df["Pileup_sumEOOT"].apply(lambda x: np.log(x))
	df["Pileup_sumLOOT"] = df["Pileup_sumLOOT"].apply(lambda x: np.log1p(x))

	df["MMuon_etaMinusGen"] = df["MMuon_etaMinusGen"].apply(lambda x: np.arctan(x*100))
	df["MMuon_phiMinusGen"] = df["MMuon_phiMinusGen"].apply(lambda x: np.arctan(x*80))
	df["MMuon_ptRatio"] = df["MMuon_ptRatio"].apply(lambda x: np.arctan((x-1)*10))
	df["MMuon_dxy"] = df["MMuon_dxy"].apply(lambda x: np.arctan(x*150))
	df["MMuon_dxyErr"] = df["MMuon_dxyErr"].apply(lambda x: np.log1p(x))
	df["MMuon_dz"] = df["MMuon_dz"].apply(lambda x: np.arctan(x*50))
	df["MMuon_dzErr"] = df["MMuon_dzErr"].apply(lambda x: np.log(x+0.001))
	df["MMuon_ip3d"] = df["MMuon_ip3d"].apply(lambda x: np.log(x+0.001))
	df["MMuon_jetPtRelv2"] = df["MMuon_jetPtRelv2"].apply(lambda x: np.log(x+0.001))
	arr = df["MMuon_jetPtRelv2"].values
	arr[arr<=-4] = np.random.normal(loc=-6.9, scale=1, size=arr[arr<=-4].shape)
	df["MMuon_jetPtRelv2"] = arr
	df["MMuon_jetRelIso"] = df["MMuon_jetRelIso"].apply(lambda x: np.log(x+0.08))
	df["MMuon_pfRelIso04_all"] = df["MMuon_pfRelIso04_all"].apply(lambda x: np.log(x+0.00001))
	arr = df["MMuon_pfRelIso04_all"].values
	arr[arr<=-7.5] = np.random.normal(loc=-11.51, scale=1, size=arr[arr<=-7.5].shape)
	df["MMuon_pfRelIso04_all"] = arr
	df["MMuon_pfRelIso03_all"] = df["MMuon_pfRelIso03_all"].apply(lambda x: np.log(x+0.00001))
	arr = df["MMuon_pfRelIso03_all"].values
	arr[arr<=-7.5] = np.random.normal(loc=-11.51, scale=1, size=arr[arr<=-7.5].shape)
	df["MMuon_pfRelIso03_all"] = arr
	df["MMuon_pfRelIso03_chg"] = df["MMuon_pfRelIso03_chg"].apply(lambda x: np.log(x+0.00001))
	arr = df["MMuon_pfRelIso03_chg"].values
	arr[arr<=-7.5] = np.random.normal(loc=-11.51, scale=1, size=arr[arr<=-7.5].shape)
	df["MMuon_pfRelIso03_chg"] = arr
	df["MMuon_ptErr"] = df["MMuon_ptErr"].apply(lambda x: np.log(x+0.001))
	df["MMuon_sip3d"] = df["MMuon_sip3d"].apply(lambda x: np.log1p(x))
	df['MMuon_isGlobal'] = df['MMuon_isGlobal'].apply(lambda x: x + 0.1*np.random.normal())
	df['MMuon_isPFcand'] = df['MMuon_isPFcand'].apply(lambda x: x + 0.1*np.random.normal())
	df['MMuon_isTracker'] = df['MMuon_isTracker'].apply(lambda x: x + 0.1*np.random.normal())
	df['MMuon_mediumId'] = df['MMuon_mediumId'].apply(lambda x: x + 0.1*np.random.normal())
	df['MMuon_softId'] = df['MMuon_softId'].apply(lambda x: x + 0.1*np.random.normal())
	df['MMuon_softMvaId'] = df['MMuon_softMvaId'].apply(lambda x: x + 0.1*np.random.normal())
	df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
	print(df)
		
	# open hdf5 file for saving
	f = h5py.File(f'amuons{f}.hdf5','w')

	dset = f.create_dataset("data", data=df.values, dtype='f4')

	f.close()
	
