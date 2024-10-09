#ifndef EDEPHIT_H
#define EDEPHIT_H

#include <iostream>

#include "EDEPUtils.h"

#include "TG4Event.h"

class EDEPHit {

  public:
    EDEPHit(TG4HitSegment hit) : start_(hit.GetStart()), stop_(hit.GetStop()),
                                 energy_deposit_(hit.GetEnergyDeposit()), secondary_deposit_(hit.GetSecondaryDeposit()), 
                                 track_length_(hit.GetTrackLength()), contrib_(hit.Contrib[0]), primary_id_(hit.GetPrimaryId()), h_index(-1) {};

    EDEPHit(TG4HitSegment hit, int i) : start_(hit.GetStart()), stop_(hit.GetStop()),
                                 energy_deposit_(hit.GetEnergyDeposit()), secondary_deposit_(hit.GetSecondaryDeposit()), 
                                 track_length_(hit.GetTrackLength()), contrib_(hit.Contrib[0]), primary_id_(hit.GetPrimaryId()), h_index(i) {};
    ~EDEPHit() {};

    const TLorentzVector& GetStart()    const {return start_;};
    const TLorentzVector& GetStop()     const {return stop_;};
    const double& GetEnergyDeposit()    const {return energy_deposit_;};
    const double& GetSecondaryDeposit() const {return secondary_deposit_;};
    const double& GetTrackLength()      const {return track_length_;};
    const int& GetGetContrib()          const {return contrib_;};
    const int& GetPrimaryId()           const {return primary_id_;};
    const int& GetId()                  const {return h_index;};
  
  private:
   TLorentzVector start_;
   TLorentzVector stop_;
   double energy_deposit_;
   double secondary_deposit_;
   double track_length_;
   int contrib_;
   int primary_id_;
   int h_index;
};


using EDEPHitsMap = std::map<component, std::vector<EDEPHit>>;

#endif
