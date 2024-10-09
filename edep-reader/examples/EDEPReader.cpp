#include <TFile.h>
#include <TGeoManager.h>
#include <TROOT.h>
#include <TTreeReader.h>

#include <iostream>

#include "EDEPTree.h"

int main(int argc, char *argv[])
{
  // const auto do_display = false;

  // set batch mode
  gROOT->SetBatch();

  // open mc file
  // auto fMc = TFile::Open("../../numu_LBNF_GV2_prod_1.genie.edep-sim.root");
  // auto fMc = TFile::Open("../../mu_pos2_1000.edep.root");
  auto fMc = TFile::Open("../../mu_pos2_1000_4sett_2.edep.root");
  // auto fMc = TFile::Open("/storage/gpfs_data/neutrino/SAND/GRAIN-PHYSICS-CASE/prod/numu_LBNF_GV2_prod/data/numu_LBNF_GV2_prod_1.genie.edep-sim.root");DC
  // auto fMc = TFile::Open("/storage/gpfs_data/neutrino/users/dcasazza/sand-reco/e_pos2_1000.edep.root");
  // auto fMc = TFile::Open("/storage/gpfs_data/neutrino/users/dcasazza/sand-reco/mu_pos2_1000.edep.root");
  // TFile *f = TFile::Open("/storage/gpfs_data/neutrino/users/dcasazza/sand-reco/e_pos2_1000_branch4.digit.Clusters.root");
  // TFile *f = TFile::Open("/storage/gpfs_data/neutrino/users/dcasazza/sand-reco/mu_pos2_1000_branch4.digit.Clusters.root");
  
  
  
  


  // auto fMc = TFile::Open("../numu_LBNF_GV2_prod_1.genie.edep-sim.root");
  // auto fMc =
      // TFile::Open(argv[1]);
  
  // read MC TTree
  TTreeReader tMc("EDepSimEvents", fMc);
  TTreeReaderValue<TG4Event> ev(tMc, "Event");
  TGeoManager* geo = 0;
  geo = (TGeoManager*)fMc->Get("EDepSimGeometry");

  auto nev = tMc.GetEntries();
  // // auto ev_min = std::atoi(argv[1]);
  // // auto ev_max = std::atoi(argv[2]);
  auto ev_min = 0;
  auto ev_max = 1000;

  // loop over events

  EDEPTree tree;
  for (int ev_id = ev_min; ev_id < ev_max; ev_id++) {
  // for (int ev_id = ev_min; ev_id <= nev; ev_id++) {
    // get digits and mc info for event
    std::cout << "Event number: " << ev_id << std::endl; //"number of events: " << nev << std::endl; 
    tMc.SetEntry(ev_id);

    std::cout << "########### Inizializing tree from edep trajectories and hits" << std::endl;
    tree.InizializeFromEdep(*ev.Get(), geo);
    
    // std::cout << "########### Printing the tree" << std::endl;
    std::string print_string = "";
    // tree.Print(print_string);

    // LArHit, Straw, EMCalSci
    component comp = component::ECAL;
    std::cout << "########### Filtering all the trajectories Entering in " << component_to_string.at(comp) << std::endl;
    std::vector<EDEPTrajectory> filteredTrj;
    tree.Filter(std::back_insert_iterator<std::vector<EDEPTrajectory>>(filteredTrj), 
      [comp](const EDEPTrajectory& trj) { return trj.IsEntering(comp);} );
    std::cout << "Number of filtered traj: " << filteredTrj.size() << std::endl;
    // for (auto t:filteredTrj) t.Print(print_string, 0);
    filteredTrj.clear();

    // std::cout << "########### Filtering all the trajectories Exiting the " << component_to_string.at(comp) << std::endl;
    // tree.Filter(std::back_insert_iterator<std::vector<EDEPTrajectory>>(filteredTrj), 
    //   [comp](const EDEPTrajectory& trj) { return trj.IsExiting(comp);} );
    // std::cout << "Number of filtered traj: " << filteredTrj.size() << std::endl;



    // print_string = "";
    // tree.GetTrajectory(5)->Print(print_string);
    // for (const auto& e:tree.GetTrajectory(5)->GetHitMap())
    //   for (const auto& h:e.second)
    //     std::cout << component_to_string.at(e.first) << " " << h.GetId() << std::endl;


    // std::cout << tree.GetTrajectory(5)->HasHitWithIdInDetector(38,  component::GRAIN) << std::endl;
    // std::cout << tree.GetTrajectory(5)->HasHitWithIdInDetector(108, component::GRAIN) << std::endl;
    // std::cout << tree.GetTrajectory(5)->HasHitWithIdInDetector(109, component::GRAIN) << std::endl;
    // std::cout << tree.GetTrajectory(5)->HasHitWithIdInDetector(110, component::GRAIN) << std::endl;

    // std::cout << tree.GetTrajectoryWithHitIdInDetector(41, component::ECAL)->GetId() << std::endl;

    print_string = "";
    tree.Print(print_string);

    // std::cout << "########### Filtering all the trajectories with hits in argon" << std::endl;
    // // LArHit, Straw, EMCalSci
    // const char* detName = "LArHit";
    // std::vector<EDEPTrajectory> filteredTrj;
    // tree.Filter(std::back_insert_iterator<std::vector<EDEPTrajectory>>(filteredTrj), 
    //   [detName](const EDEPTrajectory& trj) { return trj.HasHitInDetector(detName);} );
    // std::cout << "Number of filtered traj: " << filteredTrj.size() << std::endl;
    // filteredTrj.clear();
    
    // std::cout << "########### Filtering all the trajectories with hits in argon and ecal" << std::endl;
    // tree.Filter(std::back_insert_iterator<std::vector<EDEPTrajectory>>(filteredTrj), 
    // [detName](const EDEPTrajectory& trj) { return (trj.HasHitInDetector("EMCalSci") && trj.HasHitInDetector("LArHit"));} );
    // std::cout << "Number of filtered traj: " << filteredTrj.size() << std::endl;
  
    // std::cout << "########### Selecting trj with id 5 looking at the full tree and printing it" << std::endl;
    // print_string = "";
    // tree.GetTrajectory(5)->Print(print_string);

    // std::cout << "Energy in LArHit " << tree.GetTrajectory(5)->GetDepositedEnergy("LArHit") << std::endl;
    // std::cout << "Energy in Straw " << tree.GetTrajectory(5)->GetDepositedEnergy("Straw") << std::endl;
    // std::cout << "Energy in EMCalSci " << tree.GetTrajectory(5)->GetDepositedEnergy("EMCalSci") << std::endl;
  
    // std::cout << "########### Selecting trj with id 23 looking only at the children of trj with id 6 and printing it" << std::endl;
    // print_string = "";
    // tree.GetTrajectoryFrom(23, tree.GetTrajectory(1))->Print(print_string);

    // std::cout << "########### Some checks for trajectories with different ids in the full tree or looking only at the children of some trjectories" << std::endl;

    // std::cout << "Is trj with id 6 in the tree? -> " << tree.HasTrajectory(6) << std::endl;
    // std::cout << "Is trj with id 4564654 in the tree? -> " << tree.HasTrajectory(4564654) << std::endl;
    // std::cout << "Is trj with id 25 in the trajectory with id 6? -> " << tree.IsTrajectoryIn(25, tree.GetTrajectory(6)) << std::endl;
    // std::cout << "Is trj with id 1 in the trajectory with id 6? -> " << tree.IsTrajectoryIn(1, tree.GetTrajectory(6)) << std::endl;
    
    // std::cout << "########### Selecting the trajectory having a child with id 23" << std::endl;
    // print_string = "";
    // tree.GetParentOf(23)->Print(print_string);
    // std::cout << "########### Selecting the trajectory having a child with id 25 only if it is a child of trajectory 6" << std::endl;
    // print_string = "";
    // tree.GetParentOf(25, tree.GetTrajectory(6))->Print(print_string);

    // std::cout << "########### Removing trajectory 3" << std::endl;
    // tree.RemoveTrajectory(3);
    // std::cout << "########### Removing trajectory 25 only if it is a child of trajectory 6" << std::endl;
    // tree.RemoveTrajectoryFrom(25, tree.GetTrajectory(6));

    // std::cout << "########### Moving trajectory 6 to be child of trajectory 8" << std::endl;
    // tree.MoveTrajectoryTo(6, 8);
    // std::cout << "########### Printing the tree after the changes" << std::endl;
    // print_string = "";
    // tree.Print(print_string);

  }

    
  return 0;
}