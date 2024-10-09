#include <vector>
#include <utility>
#include <string>
#include <complex>
#include <tuple>
#include <iostream>
#include <fstream>
#include "TLeaf.h"
#include "TFile.h"
#include "TTree.h"

#include "struct.h"
#include "EDEPUtils.h"
#include "Linkdef.h"

int Testing2(std::string input, std::string input2) {
    const char* finname = input.c_str();
    const char* finname2 = input2.c_str();
    Double_t DEArr[10000];
    Double_t DEArrClus[10000];
    TFile f(finname, "READ");
    TTree* tCluster = (TTree*)f.Get("tCluster");
    TTreeReader cluReader(tCluster);
    TTreeReaderArray<cluster> clu_vec = { cluReader, "cluster" };
    int pp = 0;

    TH1F* ClusterEnergy = new TH1F("ClusterEnergy", "Total Energy reconstructed in the ECal;Energy [MeV];Entries/bins", 50, 0., 500.);

    while (cluReader.Next()) {
        double eClu_sum = 0;
        int qq = 0;
        for (const cluster& clu : clu_vec) {
            qq++;
            cout << "Cluster " << qq << " Energia: " << clu.e << endl;
            eClu_sum += clu.e;
        }
        DEArrClus[pp]= eClu_sum;
        ClusterEnergy->Fill(eClu_sum);
        pp++;
    }
    const double m_mu = 106;
    const double m_e = 0.51;
    TH1F* depEnergy = new TH1F("depEnergy", "Energy deposited by the particles in the ECal (Act+Pass);Energy [MeV];Entries/bins", 50, 0., 500.);
    TH1F* SegDetEnergy = new TH1F("SegDetEnergy", "Total Energy from SegmentDetectors;Energy [MeV];Entries/bins", 50, 0., 500.);
    auto fmc = TFile::Open(finname2);
    TTreeReader tMc("EDepSimEvents", fmc);
    TTreeReaderValue<TG4Event> ev(tMc, "Event");
    TGeoManager* geo = 0;
    geo = (TGeoManager*)fmc->Get("EDepSimGeometry");
    double x = 0, y = 0, z = 0;
    int zz = 0;
    while (tMc.Next()) {
        double in_energy = 0, dep_energy = 0;
        double prevEn = 0;
        for (auto const& h : ev->Trajectories) {
            int first = 0;
            for (auto const& f : h.Points) {
                //cout << "///////////////////" << endl;
                TLorentzVector position = f.GetPosition();
                const TVector3 momentum = f.GetMomentum();
                TGeoNode* node = geo->FindNode(position.X(), position.Y(), position.Z());
                std::string volume = node->GetName();
                std::string active = "ECALActive";
                std::string	passive = "ECALPassive";
                /*if (h.GetPDGCode() == 13 && h.GetParentId()==-1) {
                    double mass = m_mu;
                    double Energy = sqrt(momentum.Px() * momentum.Px() + momentum.Py() * momentum.Py() + momentum.Pz() * momentum.Pz() + mass * mass);
                    if (volume.find(active) != std::string::npos || volume.find(passive) != std::string::npos) {
                        if (first == 0) {
                            in_energy = prevEn;
                            first = 1;
                        }
                        dep_energy = Energy;
                    }
                    prevEn = Energy;
                }*/
                if (h.GetPDGCode() == -11 /*&& h.GetParentId() == -1*/) {
                    double mass = m_e;
                    double Energy = sqrt(momentum.Px() * momentum.Px() + momentum.Py() * momentum.Py() + momentum.Pz() * momentum.Pz() + mass * mass);
                    if (volume.find(active) != std::string::npos || volume.find(passive) != std::string::npos) {
                        if (first == 0) {
                            //in_energy = prevEn;
                            first = 1;
                        }
                        dep_energy = Energy;
                    }
                    prevEn = Energy;
                }
            }
        }
        depEnergy->Fill(in_energy - dep_energy);
        DEArr[zz] = in_energy - dep_energy;
        double E_SD = 0;
        for (auto const& h : ev->SegmentDetectors) {
            if (h.first == "EMCalSci") {
                for (auto const& g : h.second) {
                    double de = g.GetEnergyDeposit();
                    E_SD = E_SD + de;
                }
            }
        }
        SegDetEnergy->Fill(E_SD);
        zz++;
    }
    TH2F* EMcVsECluster = new TH2F("EMcVsECluster", "Energy from MC truth vs Reconstructed Cluster Energy;Energy[MeV];Energy[MeV]", 50, 0., 500., 50, 0., 500.);
    for (int q = 0; q < zz; q++) {
        EMcVsECluster->Fill(DEArrClus[q], DEArr[q], 1);
    }
    auto J = new TCanvas();
    ClusterEnergy->SetDirectory(gROOT);
    ClusterEnergy->Draw();
    auto H = new TCanvas();
    depEnergy->SetDirectory(gROOT);
    depEnergy->Draw();
    auto I = new TCanvas();
    SegDetEnergy->SetDirectory(gROOT);
    SegDetEnergy->Draw();
    auto K = new TCanvas();
    EMcVsECluster->SetDirectory(gROOT);
    EMcVsECluster->Draw("colz");
    return 0;
}
