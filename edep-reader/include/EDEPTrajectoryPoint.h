#include <iostream>

#include "TG4Event.h"

class EDEPTrajectoryHit {

  public:
    EDEPTrajectoryHit(const TG4TrajectoryPoint& trajectory_hit) : position_(trajectory_hit.GetPosition()), momentum_(trajectory_hit.GetMomentum()),
                                                            process_(trajectory_hit.GetProcess()), sub_process_(trajectory_hit.GetSubprocess()) {};
    ~EDEPTrajectoryHit() {};

    const TLorentzVector& GetPosition()    const {return position_;};
    const TVector3&       GetMomentum()    const {return momentum_;};
    const int&    GetProcess()     const {return process_;};
    const int&    GetSubprocess()  const {return sub_process_;};
  
  private:
   TLorentzVector position_;
   TVector3 momentum_;
   int process_;
   int sub_process_;
};


using EDEPTrajectoryHits = std::vector<EDEPTrajectoryHit>;