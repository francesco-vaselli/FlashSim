import ROOT
import uproot
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nflows
import mynflow
import time
import awkward as ak
import os


class GenDS(Dataset):
	def __init__(self, df, cond_vars):
	
		y= df.loc[:, cond_vars].values
		self.y_train=torch.tensor(y,dtype=torch.float32)#.to(device)

	def __len__(self):
		return len(self.y_train)
  
	def __getitem__(self,idx):
		return self.y_train[idx]



ROOT.gInterpreter.Declare('''
auto closest_muon_dr(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> distances;
	distances.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		distances.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
			}
		}
		if (closest < 0.4){
			distances[i] = closest;
		}
	}
	return distances;
}


auto closest_muon_pt(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim, ROOT::VecOps::RVec<float> & ptm) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> pts;
	pts.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		pts.emplace_back(0.0);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				pts[i] = ptm[j];
			}
		}
	}
	return pts;
}


auto closest_muon_deta(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> detas;
	detas.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		detas.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				detas[i] = deta;
			}
		}
	}
	return detas;
}


auto closest_muon_dphi(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> dphis;
	dphis.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		dphis.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				dphis[i] = dphi;
			}
		}
	}
	return dphis;
}

auto second_muon_dr(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> distances;
	distances.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		distances.emplace_back(0.5);
		float closest = 0.4;
		float second_closest = 0.5;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				second_closest = closest;
				closest = dr;
			}
		}
		if (second_closest < 0.4){
			distances[i] = second_closest;
		}
	}
	return distances;
}


auto second_muon_pt(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim, ROOT::VecOps::RVec<float> & ptm) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> pts;
	pts.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		pts.emplace_back(0.0);
		float closest = 0.4;
		float second_closest = 0.5;
		float closest_pt = 0.0;
		float second_pt = 0.0;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				second_closest = closest;
				second_pt = closest_pt;
				closest = dr;
				closest_pt = ptm[j];
			}
		if (second_closest < 0.4){
			pts[i] = second_pt;
		}
		}
	}
	return pts;
}


auto second_muon_deta(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> detas;
	detas.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		detas.emplace_back(0.5);
		float closest = 0.4;
		float second_closest = 0.5;
		float closest_deta = 0.0;
		float second_deta = 0.0;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				second_closest = closest;
				second_deta = closest_deta;
				closest = dr;
				closest_deta = deta;
			}
		if (second_closest < 0.4){
			detas[i] = second_deta;
		}
		}
	}
	return detas;
}


auto second_muon_dphi(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> dphis;
	dphis.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		dphis.emplace_back(0.5);
		float closest = 0.4;
		float second_closest = 0.5;
		float closest_dphi = 0.0;
		float second_dphi = 0.0;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				second_closest = closest;
				second_dphi = closest_dphi;
				closest = dr;
				closest_dphi = dphi;
			}
		if (second_closest < 0.4){
			dphis[i] = second_dphi;
		}
		}
	}
	return dphis;
}

auto DeltaPhi(ROOT::VecOps::RVec<float> &Phi1, ROOT::VecOps::RVec<float> &Phi2) {
	auto size = Phi1.size();
   	ROOT::VecOps::RVec<float> dphis;
	dphis.reserve(size);
	for (size_t i = 0; i < size; i++) {
		Double_t dphi = TVector2::Phi_mpi_pi(Phi1[i]-Phi2[i]);
		dphis.emplace_back(dphi);
	}
	return dphis;
	}
auto closest_jet_dr(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> distances;
	distances.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		distances.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
			}
		}
		if (closest < 0.4){
			distances[i] = closest;
		}
	}
	return distances;
}
auto closest_jet_mass(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim, ROOT::VecOps::RVec<float> & massj) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> masses;
	masses.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		masses.emplace_back(0.0);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				masses[i] = massj[j];
			}
		}
	}
	return masses;
}
auto closest_jet_pt(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim, ROOT::VecOps::RVec<float> & ptj) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> pts;
	pts.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		pts.emplace_back(0.0);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				pts[i] = ptj[j];
			}
		}
	}
	return pts;
}
auto closest_jet_deta(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> detas;
	detas.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		detas.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				detas[i] = deta;
			}
		}
	}
	return detas;
}
auto closest_jet_dphi(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> dphis;
	dphis.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		dphis.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				dphis[i] = dphi;
			}
		}
	}
	return dphis;
}
auto BitwiseDecoder(ROOT::VecOps::RVec<int> &ints, int &bit) {
	auto size = ints.size();
   	ROOT::VecOps::RVec<float> bits;
	bits.reserve(size);
	int num = pow(2, (bit));
	for (size_t i = 0; i < size; i++) {
		Double_t bAND = ints[i] & num;
		if (bAND == num) {
			bits.emplace_back(1);
		}
		else {bits.emplace_back(0);}
	}
	return bits;
	}

auto muons_per_event(ROOT::VecOps::RVec<int> &MGM){
	int size = MGM.size();
	return size;
	}


auto charge(ROOT::VecOps::RVec<int> & pdgId) {
	auto size = pdgId.size();
   	ROOT::VecOps::RVec<float> charge;
	charge.reserve(size);
	for (size_t i = 0; i < size; i++) {
		if (pdgId[i] == -13) charge.emplace_back(-1); 
		else charge.emplace_back(+1);
	}
	return charge;
	}

void gens(std::string x){
		ROOT::EnableImplicitMT();
		ROOT::RDataFrame d("Events", x);
	// create first mask
	auto d_def = d.Define("MuonMask", "Muon_genPartIdx >=0").Define("MatchedGenMuons", "Muon_genPartIdx[MuonMask]")
		//	.Define("JetMask","Jet_genJetIdx >=0  && Jet_genJetIdx < nGenJet").Define("MatchedGenJets","Jet_genJetIdx[JetMask]")
			.Define("MuonMaskJ", "(GenPart_pdgId == 13 | GenPart_pdgId == -13)&&((GenPart_statusFlags & 8192) > 0)") ;
	
	auto d_matched = d_def
				//.Define("MGenJet_eta", "Take(GenJet_eta,MatchedGenJets)")
				//.Define("MGenJet_mass", "Take(GenJet_mass,MatchedGenJets)")
				//.Define("MGenJet_phi", "Take(GenJet_phi,MatchedGenJets)")
				//.Define("MGenJet_pt", "Take(GenJet_pt,MatchedGenJets)")
				//.Define("MGenJet_partonFlavour", "Take(GenJet_partonFlavour,MatchedGenJets)")
				//.Define("MGenJet_hadronFlavour", "Take(GenJet_hadronFlavour,MatchedGenJets)")
				//.Alias("MGenJet_eta", "GenJet_eta")
				.Define("MMuon_pt", "GenPart_pt[MuonMaskJ]")
				.Define("MMuon_eta", "GenPart_eta[MuonMaskJ]")
				.Define("MMuon_phi", "GenPart_phi[MuonMaskJ]")
				.Define("MClosestMuon_dr", closest_muon_dr, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_deta", closest_muon_deta, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_dphi", closest_muon_dphi, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_pt", closest_muon_pt, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MSecondClosestMuon_dr", second_muon_dr, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_deta", second_muon_deta, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_dphi", second_muon_dphi, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_pt", second_muon_pt, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MGenMuon_eta", "GenPart_eta[MuonMaskJ]")
				.Define("MGenMuon_pdgId", "GenPart_pdgId[MuonMaskJ]")
				.Define("MGenMuon_charge", charge, {"MGenMuon_pdgId"})
				.Define("MGenMuon_phi", "GenPart_phi[MuonMaskJ]")
				.Define("MGenMuon_pt", "GenPart_pt[MuonMaskJ]")
				//.Define("nGenMuons", "Take(nGenPart, MuonMaskJ)")
				.Define("MGenPart_statusFlags","GenPart_statusFlags[MuonMaskJ]")
				.Define("MGenPart_statusFlags0", [](ROOT::VecOps::RVec<int> &ints){ int bit = 0; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags1", [](ROOT::VecOps::RVec<int> &ints){ int bit = 1; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags2", [](ROOT::VecOps::RVec<int> &ints){ int bit = 2; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags3", [](ROOT::VecOps::RVec<int> &ints){ int bit = 3; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags4", [](ROOT::VecOps::RVec<int> &ints){ int bit = 4; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags5", [](ROOT::VecOps::RVec<int> &ints){ int bit = 5; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags6", [](ROOT::VecOps::RVec<int> &ints){ int bit = 6; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags7", [](ROOT::VecOps::RVec<int> &ints){ int bit = 7; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags8", [](ROOT::VecOps::RVec<int> &ints){ int bit = 8; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags9", [](ROOT::VecOps::RVec<int> &ints){ int bit = 9; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags10", [](ROOT::VecOps::RVec<int> &ints){ int bit = 10; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags11", [](ROOT::VecOps::RVec<int> &ints){ int bit = 11; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags12", [](ROOT::VecOps::RVec<int> &ints){ int bit = 12; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags13", [](ROOT::VecOps::RVec<int> &ints){ int bit = 13; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags14", [](ROOT::VecOps::RVec<int> &ints){ int bit = 14; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MClosestJet_dr", closest_jet_dr, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi"})
				.Define("MClosestJet_deta", closest_jet_deta, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi"})
				.Define("MClosestJet_dphi", closest_jet_dphi, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi"})
				.Define("MClosestJet_pt", closest_jet_pt, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi", "GenJet_pt"})
				.Define("MClosestJet_mass", closest_jet_mass, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi", "GenJet_mass"});
	
	vector<string> col_to_save = 
		{"nGenJet", "nElectron", "MClosestMuon_dr", "MClosestMuon_pt", "MClosestMuon_deta", "MClosestMuon_dphi", "MSecondClosestMuon_dr", "MSecondClosestMuon_pt",
			"MSecondClosestMuon_deta", "MSecondClosestMuon_dphi", "GenJet_eta", "GenJet_mass", "GenJet_phi", "GenJet_pt", "GenJet_partonFlavour", "GenJet_hadronFlavour",
			"MGenMuon_eta" , "MGenMuon_phi", "MGenMuon_pt", "MGenMuon_charge", "MGenPart_statusFlags0", "MGenPart_statusFlags1", "MGenPart_statusFlags2", "MGenPart_statusFlags3", "MGenPart_statusFlags4",	
			"MGenPart_statusFlags5", "MGenPart_statusFlags6", "MGenPart_statusFlags7", "MGenPart_statusFlags8", "MGenPart_statusFlags9", "MGenPart_statusFlags10", 
			"MGenPart_statusFlags11",	"MGenPart_statusFlags12", "MGenPart_statusFlags13", "MGenPart_statusFlags14", "MClosestJet_dr", "MClosestJet_deta", 
			"MClosestJet_dphi", "MClosestJet_pt", "MClosestJet_mass",	"Pileup_gpudensity", "Pileup_nPU", "Pileup_nTrueInt", "Pileup_pudensity", "Pileup_sumEOOT", 
			"Pileup_sumLOOT", "event", "run", "Electron_eta", "Electron_pt", "Electron_mvaFall17V2Iso_WP90"};	
	
	//d_matched.Snapshot("GensJ", "testGensJ.root", col_to_save);
	// as of now this will get saved and overwritten in the execution folder
	d_matched.Snapshot("Gens", "testGens.root", col_to_save);
}
''')

UPSAMPLE_FACTOR=10000
START=0
STOP=2

def nbd(jet_model, muon_model, root, file_path, new_root):
	
	# select nano aod, process and save intermmediate files to disk
	s = str(os.path.join(root, file_path))
	ROOT.gens(s)
	print('done saving intermidiate file')
	
	muon_cond = ["MGenMuon_eta" , "MGenMuon_phi", "MGenMuon_pt", "MGenMuon_charge", "MGenPart_statusFlags0", "MGenPart_statusFlags1", "MGenPart_statusFlags2", "MGenPart_statusFlags3", 
				"MGenPart_statusFlags4","MGenPart_statusFlags5", "MGenPart_statusFlags6", "MGenPart_statusFlags7", "MGenPart_statusFlags8", "MGenPart_statusFlags9", "MGenPart_statusFlags10", 
			"MGenPart_statusFlags11",	"MGenPart_statusFlags12", "MGenPart_statusFlags13", "MGenPart_statusFlags14", "MClosestJet_dr", "MClosestJet_deta", 
			"MClosestJet_dphi", "MClosestJet_pt", "MClosestJet_mass",	"Pileup_gpudensity", "Pileup_nPU", "Pileup_nTrueInt", "Pileup_pudensity", "Pileup_sumEOOT", 
			"Pileup_sumLOOT"]
	jet_cond = ["MClosestMuon_dr", "MClosestMuon_pt", "MClosestMuon_deta", "MClosestMuon_dphi", "MSecondClosestMuon_dr", "MSecondClosestMuon_pt",
			"MSecondClosestMuon_deta", "MSecondClosestMuon_dphi", "GenJet_eta", "GenJet_mass", "GenJet_phi", "GenJet_pt", "GenJet_partonFlavour", "GenJet_hadronFlavour",]
	# read processed files for jets and save event structure
	tree = uproot.open("testGens.root:Gens", num_workers=20)
	df = tree.arrays(jet_cond, library="pd", entry_start=START, entry_stop=STOP).astype('float32').dropna()
	#numb_ev = np.arange(0, (df.index.get_level_values(0).values[-1]+1)*UPSAMPLE_FACTOR)
	#numb_ev = np.arange(0, UPSAMPLE_FACTOR)
#	numb_ev = np.repeat(df.index.get_level_values(0).values, UPSAMPLE_FACTOR)
	#numb_sub_ev = np.arange(0,df.index.get_level_values(1).values[-1]+1)
#	print(len(df.index.get_level_values(1).values))
	#numb_sub_ev = np.concatenate([df.index.get_level_values(1).values]*UPSAMPLE_FACTOR)
#	print(numb_sub_ev)
#	up_index = pd.MultiIndex.from_product([numb_ev, numb_sub_ev],names=['event', 'object'])
#	print(numb_ev, numb_sub_ev, df)
	df = pd.concat([df]*UPSAMPLE_FACTOR, axis=0)#.set_index(up_index)	
	#numb_sub_ev = df.index.get_level_values(1).values
	#print(len(numb_sub_ev), len(numb_ev))
	#up_index = pd.MultiIndex.from_arrays([numb_ev, numb_sub_ev],names=['event', 'object'])
	up_index = df.index.set_levels(np.arange(0, len(df)+1), level=0)
	df = df.set_index(up_index)	
	
	print(df.index)
	df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
	# print((df['nGenMuons'] == 0).any())
	# in short: this is a better way of keeping track of substructure with multiindex
	# for jets is the ONLY way as genjet is not accurate (different from our mask?)
	jets_ev_index = np.unique(df.index.get_level_values(0).values)
	events_structure_jets = df.reset_index(level=1).index.value_counts().sort_index().values
	print(events_structure_jets)
	print(len(events_structure_jets))
	print(sum(events_structure_jets))
	
	# reset dataframe index fro 1to1 gen
	df.reset_index(drop=True)
	
	dfm = tree.arrays(muon_cond, library="pd", entry_start=START, entry_stop=STOP).astype('float32').dropna()
	#numb_ev = np.arange(0, dfm.index.get_level_values(0).values[-1]*UPSAMPLE_FACTOR)
	numb_sub_ev = np.arange(0,dfm.index.get_level_values(1).values[-1]+1)
	up_index = pd.MultiIndex.from_product([numb_ev, numb_sub_ev],names=['event', 'object'])
	dfm = pd.concat([dfm]*UPSAMPLE_FACTOR, axis=0).set_index(up_index)	
	dfm = dfm[~dfm.isin([np.nan, np.inf, -np.inf]).any(1)]
	phys_pt = dfm["MGenMuon_pt"].values # for later rescaling
	print(phys_pt.shape) 
	dfm["MGenMuon_pt"] = dfm["MGenMuon_pt"].apply(lambda x: np.log(x)) # for conditioning
	dfm["MClosestJet_pt"] = dfm["MClosestJet_pt"].apply(lambda x: np.log1p(x))
	dfm["MClosestJet_mass"] = dfm["MClosestJet_mass"].apply(lambda x: np.log1p(x))
	dfm["Pileup_sumEOOT"] = dfm["Pileup_sumEOOT"].apply(lambda x: np.log(x))
	dfm["Pileup_sumLOOT"] = dfm["Pileup_sumLOOT"].apply(lambda x: np.log1p(x))
	dfm = dfm[~dfm.isin([np.nan, np.inf, -np.inf]).any(1)]
	print(dfm)
	muons_ev_index = np.unique( dfm.index.get_level_values(0).values)
	print(muons_ev_index)
	events_structure_muons = dfm.reset_index(level=1).index.value_counts().sort_index().values
	print(len(events_structure_muons))
	print(sum(events_structure_muons))
	#dfg = dfg.loc[df.index.get_level_values(0)]

	# reset dataframe index fro 1to1 gen
	dfm.reset_index(drop=True)
	charges = np.reshape(dfm['MGenMuon_charge'].values, (len(dfm['MGenMuon_charge'].values), 1))
	# we now condition on charge as well, so it should not be removed anymore
	# dfm = dfm.drop(columns=['MGenMuon_charge'])
	# muon_cond.remove('MGenMuon_charge')	
		
	dfe = tree.arrays(['event', 'run'], library="pd", entry_start=START, entry_stop=STOP).astype(np.longlong).dropna()
	#numb_ev = np.arange(1,dfe.index.get_level_values(0).values[-1]*UPSAMPLE_FACTOR+1)
	numb_ev = np.arange(1,UPSAMPLE_FACTOR+1)
	up_index = pd.Index(numb_ev)
	dfe = pd.concat([dfe]*UPSAMPLE_FACTOR, axis=0).set_index(up_index)	
	print(dfe)
	print(f"Total number of events is {len(dfe)}") 
	dfe = dfe[~dfe.isin([np.nan, np.inf, -np.inf]).any(1)]
	events_structure = dfe.values
	print(events_structure.shape, events_structure.shape)

	dfel = tree.arrays(['Electron_mvaFall17V2Iso_WP90', 'Electron_eta', 'Electron_pt'], library="pd", entry_start=START, entry_stop=STOP).astype('float32').dropna()
	#numb_ev = np.arange(0, dfel.index.get_level_values(0).values[-1]*UPSAMPLE_FACTOR)
	numb_ev = np.arange(0, UPSAMPLE_FACTOR)
	numb_sub_ev = np.arange(0,dfel.index.get_level_values(1).values[-1]+1)
	up_index = pd.MultiIndex.from_product([numb_ev, numb_sub_ev],names=['event', 'object'])
	dfel = pd.concat([dfel]*UPSAMPLE_FACTOR, axis=0).set_index(up_index)	
	print(dfel)
	dfel = dfel[~dfel.isin([np.nan, np.inf, -np.inf]).any(1)]
	el_ev_index = np.unique( dfel.index.get_level_values(0).values)
	print(el_ev_index)
	events_structure_el = dfel.reset_index(level=1).index.value_counts().sort_index().values
	to_ttreel = dfel.values
	print(events_structure_el.shape, events_structure_el.shape)

	# adjust structure if some events have no jets	
	zeros = np.zeros(len(dfe), dtype=int)
	print(len(jets_ev_index), len(events_structure_jets))
	np.put(zeros, jets_ev_index, events_structure_jets, mode='rise')
	events_structure_jets = zeros
	print(events_structure_jets.shape, events_structure_jets)
	print(sum(events_structure_jets))
	
	# adjust structure if some events have no muons	
	zeros = np.zeros(len(dfe), dtype=int)
	print(len(muons_ev_index), len(events_structure_muons))
	np.put(zeros, muons_ev_index, events_structure_muons, mode='rise')
	events_structure_muons = zeros
	print(events_structure_muons.shape, events_structure_muons)
	print(sum(events_structure_muons))
	
	# adjust structure if some events have no electrons
	zeros = np.zeros(len(dfe), dtype=int)
	np.put(zeros, el_ev_index, events_structure_el, mode='rise')
	events_structure_el = zeros
	print(events_structure_el.shape, events_structure_el)
	print(sum(events_structure_el))
	
	jet_dataset = GenDS(df, jet_cond)
	muon_dataset = GenDS(dfm, muon_cond)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
	batch_size = 10000
	jet_loader = DataLoader(jet_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,  num_workers=20)
	flow = jet_model
	# flow, _, _, _, trh, tsh = mynflow.load_model("../nflows/saves", "model_jets_plus_muons_@epoch_180.pt")
	tot_sample = []
	leftover_sample = []
	times = []
	leftover_shape = None 

	for batch_idx, y in enumerate(jet_loader):

		if device is not None:
			y = y.float().to(device, non_blocking=True)

    # Compute log prob
		# print(y.shape)
		if len(y) == batch_size:
			start = time.time()
			sample = flow.sample(1, context=y)
			taken = time.time() - start
			print(f'Done {batch_size} data in {taken}s')
			times.append(taken)
			sample = sample.detach().cpu().numpy()
			sample = np.squeeze(sample, axis=1)
	    		#print(sample.shape)
			tot_sample.append(sample)
		
		else:
			leftover_shape = len(y)
			sample = flow.sample(1, context=y)
			sample = sample.detach().cpu().numpy()
			sample = np.squeeze(sample, axis=1)
	    # print(sample.shape)
			leftover_sample.append(sample)



	print(np.mean(times))
	tot_sample = np.array(tot_sample)
	if leftover_shape != None:
		tot_sample = np.reshape(tot_sample, ((len(jet_loader)-1)*batch_size, 17))
		leftover_sample = np.array(leftover_sample)
		leftover_sample = np.reshape(leftover_sample, (leftover_shape, 17))
		totalj = np.concatenate((tot_sample, leftover_sample), axis=0)
	if leftover_shape == None:
		tot_sample = np.reshape(tot_sample, ((len(jet_loader))*batch_size, 17))
		totalj = tot_sample 
		

	# correct ratios and differences for jets (no preprocessing)
	totalj[:, 7] = totalj[:, 7] + df['GenJet_eta'].values
	totalj[:, 9] = totalj[:, 9] * df['GenJet_mass'].values
	totalj[:, 11] = totalj[:, 11] +  df['GenJet_phi'].values
	totalj[:, 11]= np.where( totalj[:, 11]< -np.pi, totalj[:, 11] + 2*np.pi, totalj[:, 11])
	totalj[:, 11]= np.where( totalj[:, 11]> np.pi, totalj[:, 11] - 2*np.pi, totalj[:, 11])
	totalj[:, 12] = totalj[:, 12] * df['GenJet_pt'].values

	# transform back dequantized flags
	totalj[:, 15]= np.rint(totalj[:, 15])
	totalj[:, 16]= np.rint(totalj[:, 16])

	print(totalj.shape)
	
	
	muon_loader = DataLoader(muon_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,  num_workers=20)
	flow = muon_model
	# flow, _, _, _, trh, tsh = mynflow.load_model("../nflows/saves", "model_muons_final_@epoch_430.pt")
	tot_sample = []
	leftover_sample = []
	times = []
	leftover_shape_muon = None
	print('now generating muons')
	for batch_idx, y in enumerate(muon_loader):

		if device is not None:
			y = y.float().to(device, non_blocking=True)

    # Compute log prob
		# print(y.shape)
		if len(y) == batch_size:
			start = time.time()
			sample = flow.sample(1, context=y)
			taken = time.time() - start
			print(f'Done {batch_size} data in {taken}s')
			times.append(taken)
			sample = sample.detach().cpu().numpy()
			sample = np.squeeze(sample, axis=1)
	    # print(sample.shape)
			tot_sample.append(sample)
		
		else:
			leftover_shape_muon = len(y)
			sample = flow.sample(1, context=y)
			sample = sample.detach().cpu().numpy()
			sample = np.squeeze(sample, axis=1)
	    # print(sample.shape)
			leftover_sample.append(sample)


	print(np.mean(times))
	tot_sample = np.array(tot_sample)
	if leftover_shape_muon != None:
		tot_sample = np.reshape(tot_sample, ((len(muon_loader)-1)*batch_size, 22))
		leftover_sample = np.array(leftover_sample)
		leftover_sample = np.reshape(leftover_sample, (leftover_shape, 22))
		totalm = np.concatenate((tot_sample, leftover_sample), axis=0)
	if leftover_shape_muon == None:
		tot_sample = np.reshape(tot_sample, ((len(muon_loader))*batch_size, 22))
		totalm = tot_sample
	
	
	muon_names = ["etaMinusGen", "phiMinusGen", "ptRatio", "dxy", "dxyErr",  "dz", "dzErr", "ip3d", "isGlobal", "isPFcand","isTracker", 
					"jetPtRelv2","jetRelIso", "mediumId", "pfRelIso03_all", "pfRelIso03_chg", "pfRelIso04_all", 
					"ptErr","sip3d", "softId", "softMva", "softMvaId"]

	# transform muons back to physical distributions
	df = pd.DataFrame(data=totalm, columns=['MMuon_'+s for s in muon_names])
	df["MMuon_etaMinusGen"] = df["MMuon_etaMinusGen"].apply(lambda x: np.tan(x)/100)
	df["MMuon_phiMinusGen"] = df["MMuon_phiMinusGen"].apply(lambda x: np.tan(x)/80)
	df["MMuon_ptRatio"] = df["MMuon_ptRatio"].apply(lambda x: np.tan(x)/10 + 1)#np.tan((x/10)+1))
	df["MMuon_dxy"] = df["MMuon_dxy"].apply(lambda x: np.tan(x)/150)
	df["MMuon_dxyErr"] = df["MMuon_dxyErr"].apply(lambda x: np.exp(x)-1)
	df["MMuon_dz"] = df["MMuon_dz"].apply(lambda x: np.tan(x)/50)
	df["MMuon_dzErr"] = df["MMuon_dzErr"].apply(lambda x: np.exp(x)-0.001)
	df["MMuon_ip3d"] = df["MMuon_ip3d"].apply(lambda x: np.exp(x)-0.001)
	arr = df["MMuon_jetPtRelv2"].values
	arr = np.where(arr<=-4, -6.9, arr)
	df["MMuon_jetPtRelv2"] = arr
	df["MMuon_jetPtRelv2"] = df["MMuon_jetPtRelv2"].apply(lambda x: np.exp(x)-0.001)
	df["MMuon_jetRelIso"] = df["MMuon_jetRelIso"].apply(lambda x: np.exp(x)-0.08)

	arr = df["MMuon_pfRelIso04_all"].values
	arr = np.where(arr<=-7.5, -11.51, arr)
	df["MMuon_pfRelIso04_all"] = arr
	df["MMuon_pfRelIso04_all"] = df["MMuon_pfRelIso04_all"].apply(lambda x: np.exp(x)-0.00001)

	arr = df["MMuon_pfRelIso03_all"].values
	arr = np.where(arr<=-7.5, -11.51, arr)
	df["MMuon_pfRelIso03_all"] = arr
	df["MMuon_pfRelIso03_all"] = df["MMuon_pfRelIso03_all"].apply(lambda x: np.exp(x)-0.00001)

	arr = df["MMuon_pfRelIso03_chg"].values
	arr = np.where(arr<=-7.5, -11.51, arr)
	df["MMuon_pfRelIso03_chg"] = arr
	df["MMuon_pfRelIso03_chg"] = df["MMuon_pfRelIso03_chg"].apply(lambda x: np.exp(x)-0.00001)

	df["MMuon_ptErr"] = df["MMuon_ptErr"].apply(lambda x: np.exp(x)-0.001)
	df["MMuon_sip3d"] = df["MMuon_sip3d"].apply(lambda x: np.exp(x)-1)

	arr = df["MMuon_isGlobal"].values
	arr = np.where(arr<=0.5, 0, 1)
	df['MMuon_isGlobal'] = arr

	arr = df["MMuon_isPFcand"].values
	arr = np.where(arr<=0.5, 0, 1)
	df['MMuon_isPFcand'] = arr

	arr = df["MMuon_isTracker"].values
	arr = np.where(arr<=0.5, 0, 1)
	df['MMuon_isTracker'] = arr

	arr = df["MMuon_mediumId"].values
	arr = np.where(arr<=0.5, 0, 1)
	df['MMuon_mediumId'] = arr
	arr = df["MMuon_softId"].values
	arr = np.where(arr<=0.5, 0, 1)
	df['MMuon_softId'] = arr

	arr = df["MMuon_softMvaId"].values
	arr = np.where(arr<=0.5, 0, 1)
	df['MMuon_softMvaId'] = arr

	# df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)] 
	# for now I'm removing this check,it shouldn't be needed and it would mess up the event structure
	totalm  = df.values

	# correct ratios and differences for muons 
	# (we have preprocessing so it has to be done now)
	totalm[:, 0] = totalm[:, 0] + dfm['MGenMuon_eta'].values
	totalm[:, 1] = totalm[:, 1] +  dfm['MGenMuon_phi'].values
	totalm[:, 1]= np.where( totalm[:, 1]< -np.pi, totalm[:, 1] + 2*np.pi, totalm[:, 1])
	totalm[:, 1]= np.where( totalm[:, 1]> np.pi, totalm[:, 1] - 2*np.pi, totalm[:, 1])
	totalm[:, 2] = totalm[:, 2] * phys_pt 
	print(totalm.shape)
	
	# add charge...
	totalm = np.concatenate((totalm, charges), axis=1)

	# convert to akw arrays for saving to file
	jet_names = ["area", "btagCMVA", "btagCSVV2", "btagDeepB", "btagDeepC", 
			"btagDeepFlavB", "btagDeepFlavC", "eta", "bRegCorr", "mass", "nConstituents", "phi", "pt", "qgl",
        	"muEF", "puId", "jetId"]
	to_ttreej = dict(zip(jet_names, totalj.T))
	to_ttreej = ak.unflatten(ak.Array(to_ttreej), events_structure_jets)

	muon_names = ["eta", "phi", "pt", "dxy", "dxyErr",  "dz", "dzErr", "ip3d", "isGlobal", "isPFcand","isTracker", "jetPtRelv2","jetRelIso", 
					"mediumId", "pfRelIso03_all", "pfRelIso03_chg", "pfRelIso04_all", "ptErr","sip3d", "softId", "softMva", "softMvaId", "charge"]
	to_ttreem = dict(zip(muon_names, totalm.T))
	to_ttreem = ak.Array(to_ttreem)
	to_ttreem = ak.unflatten(to_ttreem, events_structure_muons)
	
	to_ttreee = dict(zip(['event', 'run'], events_structure.T))	
	to_ttreee = ak.Array(to_ttreee)	
	
	to_ttreel = dict(zip(['Electron_mvaFall17V2Iso_WP90', 'Electron_eta', 'Electron_pt'], to_ttreel.T))	
	to_ttreel = ak.Array(to_ttreel)	
	to_ttreel = ak.unflatten(to_ttreel, events_structure_el)

	new_path = str(os.path.join(new_root, file_path))
	new_path = os.path.splitext(new_path)[0]
	with uproot.recreate(f"{new_path}_synth_upsampled.root") as file:
		file["Events"] = {'Jet': to_ttreej, 'Muon': to_ttreem, 'Electron': to_ttreel, 'event': to_ttreee.event, 'run': to_ttreee.run}

	return
