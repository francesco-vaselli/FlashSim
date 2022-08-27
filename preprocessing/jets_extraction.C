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



auto closest_muon_dr(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, ROOT::VecOps::RVec<float> & phim) {
	
	/* Calculates the DeltaR from the closest muon object,
		if none present within 0.4, sets DR to 0.4
	*/	

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

	/* Calculates the pt of the closest muon object,
		if none present within 0.4, sets DR to 0.4 and pt to 0 GeV
	*/

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

	/* Calculates the DeltaEta of the closest muon object,
		if none present within 0.4, sets DR to 0.4 and DeltaEta to 0.5 
	*/
		
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

	/* Calculates the DeltaPhi of the closest muon object,
		if none present within 0.4, sets DR to 0.4 and DeltaPhi to 0.5 
	*/	

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
		
	/* Calculates the DeltaR from the second closest muon object,
		if none present within 0.4, sets DR to 0.4
	*/	

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

	/* Calculates the pt of the second closest muon object,
		if none present within 0.4, sets DR to 0.4 and pt to 0 GeV
	*/
	
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

	/* Calculates the DeltaEta of the second closest muon object,
		if none present within 0.4, sets DR to 0.4 and DeltaEta to 0.5 
	*/
			
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

	/* Calculates the DeltaPhi of the second closest muon object,
		if none present within 0.4, sets DR to 0.4 and DeltaPhi to 0.5 
	*/
			
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


void jets_extraction(){

	/* The main function. Uses ROOT::RDataFrame to select only jets matching to a GenJet, 
		then extracts all the conditioning variables of the GenJet and the target variables of the reco jet
		in a single row (each row contains all the conditioning and target variables for exactly one jet)
	*/

	// enable multithreading, open file and init rdataframe
	// BECAUSE OF MT ORIGINAL ORDERING OF FILE IS NOT PRESERVED
	ROOT::EnableImplicitMT();
	TFile *f =TFile::Open("root://cmsxrootd.fnal.gov///store/mc/RunIIAutumn18NanoAODv6/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/250000/047F4368-97D4-1A4E-B896-23C6C72DD2BE.root");
	ROOT::RDataFrame d("Events",f);

	// create first mask 
	auto d_def = d.Define("JetMask","Jet_genJetIdx >=0  && Jet_genJetIdx < nGenJet").Define("MatchedGenJets","Jet_genJetIdx[JetMask]")
					.Define("MuonMask", "GenPart_pdgId == 13 | GenPart_pdgId == -13") ;
	
	// A stupid print check
	auto colType1 = d_def.GetColumnType("JetMask");
	// Print column type
	std::cout << "Column JetMask" << " has type " << colType1 << std::endl;

	// The main rdataframe, LAZILY defining all the new processed columns.
	// Notice that muon conditioning variables are defined starting from GenMuons
	// Lines commented out are variables missing in some NanoAODs
	auto d_matched = d_def.Define("MGenJet_eta", "Take(GenJet_eta,MatchedGenJets)")
				.Define("MGenJet_mass", "Take(GenJet_mass,MatchedGenJets)")
				.Define("MGenJet_phi", "Take(GenJet_phi,MatchedGenJets)")
				.Define("MGenJet_pt", "Take(GenJet_pt,MatchedGenJets)")
				.Define("MGenJet_partonFlavour", "Take(GenJet_partonFlavour,MatchedGenJets)")
				.Define("MGenJet_hadronFlavour", "Take(GenJet_hadronFlavour,MatchedGenJets)")
				.Define("MMuon_pt", "GenPart_pt[MuonMask]")
				.Define("MMuon_eta", "GenPart_eta[MuonMask]")
				.Define("MMuon_phi", "GenPart_phi[MuonMask]")
				.Define("MClosestMuon_dr", closest_muon_dr, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_deta", closest_muon_deta, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_dphi", closest_muon_dphi, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_pt", closest_muon_pt, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MSecondClosestMuon_dr", second_muon_dr, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_deta", second_muon_deta, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_dphi", second_muon_dphi, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_pt", second_muon_pt, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MJet_area", "Jet_area[JetMask]")
				.Define("MJet_bRegCorr", "Jet_bRegCorr[JetMask]")
				.Define("MJet_bRegRes", "Jet_bRegRes[JetMask]")
				.Define("MJet_btagCMVA", "Jet_btagCMVA[JetMask]")
				.Define("MJet_btagCSVV2", "Jet_btagCSVV2[JetMask]")
				.Define("MJet_btagDeepB", "Jet_btagDeepB[JetMask]")
				.Define("MJet_btagDeepC", "Jet_btagDeepC[JetMask]")
				//.Define("MJet_btagDeepCvB", "Jet_btagDeepCvB[JetMask]")
				//.Define("MJet_btagDeepCvL", "Jet_btagDeepCvL[JetMask]")
				.Define("MJet_btagDeepFlavB", "Jet_btagDeepFlavB[JetMask]")
				.Define("MJet_btagDeepFlavC", "Jet_btagDeepFlavC[JetMask]")
				//.Define("MJet_btagDeepFlavCvB", "Jet_btagDeepFlavCvB[JetMask]")
				//.Define("MJet_btagDeepFlavCvL", "Jet_btagDeepFlavCvL[JetMask]")
				//.Define("MJet_btagDeepFlavQG", "Jet_btagDeepFlavQG[JetMask]")			
				//.Define("MJet_cRegCorr", "Jet_cRegCorr[JetMask]")			
				//.Define("MJet_cRegRes", "Jet_cRegRes[JetMask]")			
				.Define("MJet_chEmEF", "Jet_chEmEF[JetMask]")			
				//.Define("MJet_chFPV0EF", "Jet_chFPV0EF[JetMask]")			
				//.Define("MJet_chFPV1EF", "Jet_chFPV1EF[JetMask]")			
				//.Define("MJet_chFPV2EF", "Jet_chFPV2EF[JetMask]")			
				//.Define("MJet_chFPV3EF", "Jet_chFPV3EF[JetMask]")			
				.Define("MJet_chHEF", "Jet_chHEF[JetMask]")			
				.Define("MJet_cleanmask", "Jet_cleanmask[JetMask]")			
				.Define("MJet_etaMinusGen", "Jet_eta[JetMask]-MGenJet_eta")
				//.Define("MJet_hfsigmaEtaEta", "Jet_hfsigmaEtaEta[JetMask]")		
				//.Define("MJet_hfsigmaPhiPhi", "Jet_hfsigmaPhiPhi[JetMask]")			
				.Define("MJet_hadronFlavour", "Jet_hadronFlavour[JetMask]")		
				.Define("MJet_jetId", "Jet_jetId[JetMask]")
				.Define("MJet_mass", "Jet_mass[JetMask]")
				.Define("MJet_massRatio", "Jet_mass[JetMask]/MGenJet_mass")
				.Define("MJet_muEF", "Jet_muEF[JetMask]")
				.Define("MJet_muonSubtrFactor", "Jet_muonSubtrFactor[JetMask]")
				.Define("MJet_nConstituents", "Jet_nConstituents[JetMask]")
				.Define("MJet_nElectrons", "Jet_nElectrons[JetMask]")
				.Define("MJet_nMuons", "Jet_nMuons[JetMask]")
				.Define("MJet_neEmEF", "Jet_neEmEF[JetMask]")
				.Define("MJet_neHEF", "Jet_neHEF[JetMask]")
				.Define("MJet_partonFlavour", "Jet_partonFlavour[JetMask]")
				.Define("MJet_phifiltered", "Jet_phi[JetMask]")
				.Define("MJet_phiMinusGen", DeltaPhi,{"MJet_phifiltered", "MGenJet_phi"})
				.Define("MJet_ptRatio", "Jet_pt[JetMask]/MGenJet_pt")
				.Define("MJet_puId", "Jet_puId[JetMask]")
				//.Define("MJet_hfadjacentEtaStripsSize", "Jet_hfadjacentEtaStripsSize[JetMask]")
				//.Define("MJet_hfcentralEtaStripSize", "Jet_hfcentralEtaStripSize[JetMask]")
				//.Define("MJet_puIdDisc", "Jet_puIdDisc[JetMask]")
				.Define("MJet_qgl", "Jet_qgl[JetMask]")
				.Define("MJet_rawFactor", "Jet_rawFactor[JetMask]");


	// Optionally print columns names
	// auto v2 = d_matched.GetColumnNames();
	// for (auto &&colName : v2) std::cout <<"\""<< colName<<"\", ";

	// Define variables to save
	vector<string> col_to_save = 
		{"MClosestMuon_dr", "MClosestMuon_pt", "MClosestMuon_deta", "MClosestMuon_dphi", "MSecondClosestMuon_dr", "MSecondClosestMuon_pt",
		"MSecondClosestMuon_deta", "MSecondClosestMuon_dphi", "MGenJet_eta", "MGenJet_mass", "MGenJet_phi", "MGenJet_pt", "MGenJet_partonFlavour", 
		"MGenJet_hadronFlavour","MJet_area", "MJet_btagCMVA", "MJet_btagCSVV2", "MJet_btagDeepB", "MJet_btagDeepC", "MJet_btagDeepFlavB", "MJet_btagDeepFlavC",
        "MJet_etaMinusGen", "MJet_bRegCorr", "MJet_massRatio", "MJet_nConstituents", "MJet_phiMinusGen", 
        "MJet_ptRatio","MJet_qgl", "MJet_muEF", "MJet_puId", "MJet_jetId" 
		};

	// finally process columns and save to .root file
	d_matched.Snapshot("MJets", "MJetsA1.root", col_to_save);

}

