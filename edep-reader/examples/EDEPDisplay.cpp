#include <TFile.h>
#include <TGeoManager.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TTreeReader.h>
#include <TGraph.h>
#include <TMultiGraph.h>

#include <iostream>


#include "EDEPTree.h"

std::map<int, EColor> colorMap = {
  {22,    kRed}, //gamma
  {13,    kBlue}, //muon
  {2212,  kBlack}, //proton
  {2112,  kGray}, //neutron
  {111,   kMagenta}, //pi0
  {211,   kCyan}, //pi+
  {11,    kGreen}, //electron
  {0,     kYellow} //
};

int main(int argc, char *argv[])
{
  // const auto do_display = false;

  // set batch mode
  gROOT->SetBatch();

  // open mc file
  // auto fMc = TFile::Open("/storage/gpfs_data/neutrino/SAND/GRAIN-PHYSICS-CASE/prod/numu_LBNF_GV2_prod/data/numu_LBNF_GV2_prod_1.genie.edep-sim.root");
  auto fMc = TFile::Open("/pnfs/dune/persistent/users/abooth/nd-production/MicroProdN1p1/output/run-hadd/MicroProdN1p1_NDLAr_1E18_RHC.nu.hadd/EDEPSIM/0000000/0000000/MicroProdN1p1_NDLAr_1E18_RHC.nu.hadd.0000060.EDEPSIM.root");
  // auto fMc =
      // TFile::Open(argv[1]);
  
  double centerKLOE[3];
  const char* path_intreg =
    "volWorld_PV_1/rockBox_lv_PV_0/volDetEnclosure_PV_0/volArgonCubeDetector_PV_0"
    "/volArgonCubeCryostatWithTagger_PV_0/volArgonCubeCryostat_PV_0";

  TGeoManager* geo = 0;
  geo = (TGeoManager*)fMc->Get("EDepSimGeometry");

  double dummyLoc[3];

  geo->cd(path_intreg);

  dummyLoc[0] = 0.;
  dummyLoc[1] = 0.;
  dummyLoc[2] = 0.;
  geo->LocalToMaster(dummyLoc, centerKLOE);




  // read MC TTree
  TTreeReader tMc("EDepSimEvents", fMc);
  TTreeReaderValue<TG4Event> ev(tMc, "Event");

  auto nev = tMc.GetEntries();
  // // auto ev_min = std::atoi(argv[1]);
  // // auto ev_max = std::atoi(argv[2]);

  // get digits and mc info for event
  tMc.SetEntry(10);

  
  EDEPTree tree;
  std::cout << "########### Inizializing tree from edep trajectories and hits" << std::endl;
  tree.InizializeFromEdep(*ev.Get(), geo);
  std::string print_string = "";
  tree.Print(print_string);

  TCanvas* cev = new TCanvas("cev", "Event 0", 1000, 1000);
  cev->SetTitle("Event 0");

//  cev->DrawFrame(centerKLOE[2] - 2500, centerKLOE[1] - 2500, centerKLOE[2] + 2500, centerKLOE[1] + 2500, "ZY (side);[mm]; [mm]");
  cev->DrawFrame(-3000, -3000, 30000, 8000, "ZY (side);[mm]; [mm]");

  std::vector<EDEPTrajectory> filteredTrj;
  
  tree.Filter(std::back_insert_iterator<std::vector<EDEPTrajectory>>(filteredTrj), [] (const EDEPTrajectory& trj) { return (true/*trj.HasHitAfterTime(100) && trj.HasHitInDetector(component::STRAW)*/);});

  
  // Draw Trajectory Points
  for (const auto& trj:filteredTrj) {
    auto points = trj.GetTrajectoryPoints();
  std::cout << " --- Points ---- " << points.size() << std::endl ;
    TGraph* zy = new TGraph(points.size());
    if (colorMap.find(abs(trj.GetPDGCode())) != colorMap.end()) {
      zy->SetLineColor(colorMap[abs(trj.GetPDGCode())]);
    } else {
      zy->SetLineColor(colorMap[0]);
    }
    for (uint i = 0; i < points.size(); i++) {
      zy ->SetPoint(i, points[i].GetPosition().Z(), points[i].GetPosition().Y());
    }
    zy->Draw("L");
  }
  
  for (const auto& trj:filteredTrj) {
    auto hit_map = trj.GetHitMap();
    for (const auto& it:hit_map) {
      auto hits = it.second;
      TGraph* zy = new TGraph(hits.size());
      if (colorMap.find(abs(trj.GetPDGCode())) != colorMap.end()) {
        zy->SetLineColor(colorMap[abs(trj.GetPDGCode())]);
      } else {
        zy->SetLineColor(colorMap[0]);
      }

      for (uint i = 0; i < hits.size(); i++) {
        TLorentzVector mean_position = hits[i].GetStart() + hits[i].GetStop();
        mean_position *= 0.5;
        zy ->SetPoint(i, mean_position.Z(), mean_position.Y());
      }
      zy->Draw("P");
      }
  }

  cev->SaveAs("event1.png");






    // std::cout << "########### Filtering all the trajectories with hits in argon" << std::endl;
    // const char* detName = "LArHit";
    // std::vector<EDEPTrajectory> filteredTrj;
    // tree.Filter(std::back_insert_iterator<std::vector<EDEPTrajectory>>(filteredTrj), [detName](const EDEPTrajectory& trj) { return trj.HasHitInDetector(detName);} );
    // std::cout << "Number of filtered traj: " << filteredTrj.size() << std::endl;
    // filteredTrj.clear();
    
    // std::cout << "########### Filtering all the trajectories with hits in argon and ecal" << std::endl;
    // tree.Filter(std::back_insert_iterator<std::vector<EDEPTrajectory>>(filteredTrj), [detName](const EDEPTrajectory& trj) { return (trj.HasHitInDetector("EMCalSci") && trj.HasHitInDetector(detName));} );
    // std::cout << "Number of filtered traj: " << filteredTrj.size() << std::endl;
  
    // std::cout << "########### Selecting trj with id 5 looking at the full tree and printing it" << std::endl;
    // tree.GetTrajectory(5)->Print();
  
    // std::cout << "########### Selecting trj with id 23 looking only at the children of trj with id 6 and printing it" << std::endl;
    // tree.GetTrajectoryFrom(23, tree.GetTrajectory(1))->Print();

    // std::cout << "########### Some checks for trajectories with different ids in the full tree or looking only at the children of some trjectories" << std::endl;

    // std::cout << "Is trj with id 6 in the tree? -> " << tree.HasTrajectory(6) << std::endl;
    // std::cout << "Is trj with id 4564654 in the tree? -> " << tree.HasTrajectory(4564654) << std::endl;
    // std::cout << "Is trj with id 25 in the trajectory with id 6? -> " << tree.IsTrajectoryIn(25, tree.GetTrajectory(6)) << std::endl;
    // std::cout << "Is trj with id 1 in the trajectory with id 6? -> " << tree.IsTrajectoryIn(1, tree.GetTrajectory(6)) << std::endl;
    
    // std::cout << "########### Selecting the trajectory having a child with id 23" << std::endl;
    // tree.GetParentOf(23)->Print();
    // std::cout << "########### Selecting the trajectory having a child with id 25 only if it is a child of trajectory 6" << std::endl;
    // tree.GetParentOf(25, tree.GetTrajectory(6))->Print();

    // std::cout << "########### Removing trajectory 3" << std::endl;
    // tree.RemoveTrajectory(3);
    // std::cout << "########### Removing trajectory 25 only if it is a child of trajectory 6" << std::endl;
    // tree.RemoveTrajectoryFrom(25, tree.GetTrajectory(6));

    // std::cout << "########### Moving trajectory 6 to be child of trajectory 8" << std::endl;
    // tree.MoveTrajectoryTo(6, 8);
    // std::cout << "########### Printing the tree after the changes" << std::endl;
    // tree.Print();

 



    return 0;

}
