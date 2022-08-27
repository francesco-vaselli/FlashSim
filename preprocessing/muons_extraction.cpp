// Open a NanoAOD file and extract Gen-level condtioning AND reco targets for trainings
// Working fine with ROOT 6.22

auto DeltaPhi(ROOT::VecOps::RVec<float> &Phi1, ROOT::VecOps::RVec<float> &Phi2) {
	/* Calculates the DeltaPhi between two RVecs
	*/
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
	/* Calculates the DeltaR from the closest Jet object,
		if none present within 0.4, sets DR to 0.4
	*/
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
	
	/* Calculates the mass of the closest Jet object,
		if none present within 0.4, sets DR to 0.4 and mass to 0 GeV
	*/
	
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

	/* Calculates the pt of the closest Jet object,
		if none present within 0.4, sets DR to 0.4 and pt to 0 GeV
	*/
	
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

	/* Calculates the DeltaEta of the closest Jet object,
		if none present within 0.4, sets DR to 0.4 and DeltaEta to 0.5 
	*/
	
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

	/* Calculates the DeltaPhi of the closest Jet object,
		if none present within 0.4, sets DR to 0.4 and DeltaPhi to 0.5 
	*/	
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
	/* Utility function for performing bitwise decoding of
		GenPart_statusFlags conditioning variable
	*/
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


auto charge(ROOT::VecOps::RVec<int> & pdgId) {
	/* Assigns the correct charge to the muon according to its gen pdg id
	*/
	auto size = pdgId.size();
   	ROOT::VecOps::RVec<float> charge;
	charge.reserve(size);
	for (size_t i = 0; i < size; i++) {
		if (pdgId[i] == -13) charge.emplace_back(-1); 
		else charge.emplace_back(+1);
	}
	return charge;
	}



void muons_extraction(){

	/* The main function. Uses ROOT::RDataFrame to select only muons matching to a Gen-muon, 
		then extracts all the conditioning variables of the Gen muon and the target variables of the reco muon
		in a single row (each row contains all the conditioning and target variables for exactly one muon)
	*/

	// enable multithreading, open file and init rdataframe
	// BECAUSE OF MT ORIGINAL ORDERING OF FILE IS NOT PRESERVED
	ROOT::EnableImplicitMT();
	TFile *f =TFile::Open("root://cmsxrootd.fnal.gov///store/mc/RunIIAutumn18NanoAODv6/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/230000/91456D0B-2FDE-2B4F-8C7A-8E60260480CD.root");
	ROOT::RDataFrame d("Events",f);

	// create first mask
	auto d_def = d.Define("MuonMask", "Muon_genPartIdx >=0").Define("MatchedGenMuons", "Muon_genPartIdx[MuonMask]");

	// a stupid print check
	auto colType1 = d_def.GetColumnType("Pileup_gpudensity");
	// Print column type
	std::cout << "Column Pileup_gpudensity" << " has type " << colType1 << std::endl;

	// The main rdataframe, LAZILY defining all the new processed columns.
	// Notice that jet conditioning variables are defined starting from GenJets
	auto d_matched = d_def
				.Define("MGenMuon_eta", "Take(GenPart_eta,MatchedGenMuons)")
				.Define("MGenMuon_phi", "Take(GenPart_phi,MatchedGenMuons)")
				.Define("MGenMuon_pt", "Take(GenPart_pt,MatchedGenMuons)")
				.Define("MGenMuon_pdgId", "Take(GenPart_pdgId, MatchedGenMuons)")
				.Define("MGenMuon_charge", charge, {"MGenMuon_pdgId"})
				.Define("MGenPart_statusFlags","Take(GenPart_statusFlags,MatchedGenMuons)")
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
				.Define("MClosestJet_mass", closest_jet_mass, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi", "GenJet_mass"})
				.Define("MMuon_etaMinusGen", "Muon_eta[MuonMask]-MGenMuon_eta")
				.Define("MMuon_filteredphi", "Muon_phi[MuonMask]")
				.Define("MMuon_phiMinusGen", DeltaPhi, {"MMuon_filteredphi", "MGenMuon_phi"})
				.Define("MMuon_ptRatio", "Muon_pt[MuonMask]/MGenMuon_pt")
				.Define("MMuon_dxy", "Muon_dxy[MuonMask]")
				.Define("MMuon_dxyErr", "Muon_dxyErr[MuonMask]")
				.Define("MMuon_dz", "Muon_dz[MuonMask]")
				.Define("MMuon_dzErr", "Muon_dzErr[MuonMask]")
				.Define("MMuon_ip3d", "Muon_ip3d[MuonMask]")
				.Define("MMuon_isGlobal", "Muon_isGlobal[MuonMask]")
				.Define("MMuon_isPFcand", "Muon_isPFcand[MuonMask]")
				.Define("MMuon_isTracker", "Muon_isTracker[MuonMask]")
				.Define("MMuon_jetPtRelv2", "Muon_jetPtRelv2[MuonMask]")
				.Define("MMuon_jetRelIso", "Muon_jetRelIso[MuonMask]")
				.Define("MMuon_mediumId", "Muon_mediumId[MuonMask]")
				.Define("MMuon_pfRelIso03_all", "Muon_pfRelIso03_all[MuonMask]")
				.Define("MMuon_pfRelIso03_chg", "Muon_pfRelIso03_chg[MuonMask]")
				.Define("MMuon_pfRelIso04_all", "Muon_pfRelIso04_all[MuonMask]")
				.Define("MMuon_ptErr", "Muon_ptErr[MuonMask]")
				.Define("MMuon_sip3d", "Muon_sip3d[MuonMask]")
				.Define("MMuon_softId", "Muon_softId[MuonMask]")
				.Define("MMuon_softMva", "Muon_softMva[MuonMask]")
				.Define("MMuon_softMvaId", "Muon_softMvaId[MuonMask]");
				
	// optionally print columns names
	// auto v2 = d_matched.GetColumnNames();
	// for (auto &&colName : v2) std::cout <<"\""<< colName<<"\", ";

	//specify targets
	vector<string> col_to_save = 
		{"MGenMuon_eta" , "MGenMuon_phi", "MGenMuon_pt", "MGenMuon_charge","MGenPart_statusFlags0", "MGenPart_statusFlags1", "MGenPart_statusFlags2", 
		"MGenPart_statusFlags3", "MGenPart_statusFlags4",	"MGenPart_statusFlags5", "MGenPart_statusFlags6", "MGenPart_statusFlags7", "MGenPart_statusFlags8", 
		"MGenPart_statusFlags9", "MGenPart_statusFlags10", "MGenPart_statusFlags11",	"MGenPart_statusFlags12", "MGenPart_statusFlags13", "MGenPart_statusFlags14", 
		"MClosestJet_dr", "MClosestJet_deta", "MClosestJet_dphi", "MClosestJet_pt", "MClosestJet_mass",	"Pileup_gpudensity", "Pileup_nPU", "Pileup_nTrueInt", 
		"Pileup_pudensity", "Pileup_sumEOOT", "Pileup_sumLOOT","MMuon_etaMinusGen", "MMuon_phiMinusGen", "MMuon_ptRatio", "MMuon_dxy", "MMuon_dxyErr", "MMuon_dz", 
		"MMuon_dzErr", "MMuon_ip3d", "MMuon_isGlobal", "MMuon_isPFcand",	"MMuon_isTracker", "MMuon_jetPtRelv2","MMuon_jetRelIso", "MMuon_mediumId", 
		"MMuon_pfRelIso03_all", "MMuon_pfRelIso03_chg", "MMuon_pfRelIso04_all", "MMuon_ptErr",	"MMuon_sip3d", "MMuon_softId", "MMuon_softMva", "MMuon_softMvaId"};

	// finally process columns and save to .root file
	d_matched.Snapshot("MMuons", "MMuonsA9.root", col_to_save);

}

