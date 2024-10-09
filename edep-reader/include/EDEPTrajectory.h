#ifndef EDEPTRAJECTORY_H
#define EDEPTRAJECTORY_H

#include <iostream>

#include "EDEPHit.h"
#include "EDEPTrajectoryPoint.h"

#include <TGeoManager.h>

class EDEPTrajectory {
  
  public:
    // Constructors
    EDEPTrajectory() : p0_(0,0,0,0), parent_trajectory_(nullptr),
                       id_(-1), parent_id_(-99), pdg_code_(0) {};
    
    EDEPTrajectory(const TG4Trajectory& trajectory) : p0_(trajectory.GetInitialMomentum()),
                                               parent_trajectory_(nullptr) ,
                                               id_(trajectory.GetTrackId()), 
                                               parent_id_(trajectory.GetParentId()), 
                                               pdg_code_(trajectory.GetPDGCode()) {};
    
    EDEPTrajectory(const TG4Trajectory& trajectory, const TG4HitSegmentDetectors& hits, TGeoManager* geo);

    EDEPTrajectory(const EDEPTrajectory& trj);
    EDEPTrajectory(EDEPTrajectory&& trj);
    ~EDEPTrajectory() {};
    
    // Operators
    bool operator==(const EDEPTrajectory& trj);
    EDEPTrajectory& operator=(const EDEPTrajectory& trj);
    EDEPTrajectory& operator=(EDEPTrajectory&& trj);
    
    // Getters
          EDEPTrajectory* Get()       {return this;};
    const EDEPTrajectory* Get() const {return this;};

    EDEPTrajectory* GetParent() const {return parent_trajectory_;};
    
    int GetId()       const {return id_;};
    int GetDepth()    const {return depth_;};
    int GetParentId() const {return parent_id_;};
    int GetPDGCode()  const {return pdg_code_;};
    TLorentzVector GetInitialMomentum()  const {return p0_;};

          std::vector<EDEPTrajectory>& GetChildrenTrajectories()       {return children_trajectories_;};
    const std::vector<EDEPTrajectory>& GetChildrenTrajectories() const {return children_trajectories_;};

       // EDEPHitsMap& GetHitMap()       {return hit_map_;};
    const EDEPHitsMap& GetHitMap() const {return hit_map_;};
    const EDEPTrajectoryHits& GetTrajectoryPoints() const {return  trajectory_points_;};
    
    // Setters
    void SetId(int id) {id_ = id;};
    void SetDepth(int depth) {depth_ = depth;};
    void SetParentId(int parent_id) {parent_id_ = parent_id;};
    void SetParent(EDEPTrajectory* parent_trajectory) {parent_trajectory_ = parent_trajectory;};
    
    // Other
    void AddChild(const EDEPTrajectory& trajectory) {
      children_trajectories_.push_back(trajectory); 
      children_trajectories_.back().SetParent(this);
    };
    bool RemoveChildWithId(int child_id);

    void ComputeDepth();

    bool HasHits() const {return !hit_map_.empty();}
    bool HasHitInDetector(component component_name) const;
    double GetDepositedEnergy(component component_name);
    bool HasHitBeforeTime(double start_time) const;
    bool HasHitAfterTime(double stop_time) const;
    bool IsTrajectorySaturated() const;

    bool Match(std::string volume, std::initializer_list<std::string> names) const;
    bool IsEntering(component component_name) const;
    bool IsExiting(component component_name) const;

    template<typename Funct> bool HasHitWhere(Funct&& f) const {
      for(const auto& hits:hit_map_) {
        for(const auto& hit:hits.second) {
          if (f(hit)) {
            return true;
          }
        }
      }
      return false;
    }

    bool HasHitInTime(double start_time, double stop_time) const;
    bool HasHitWithIdInDetector( int id, component component_name) const;
    
    std::string Print(std::string& full_out, int depth = 100, int current_depth = 0) const;

    friend class EDEPTree;

  private:
    TLorentzVector p0_;
    EDEPHitsMap hit_map_;
    EDEPTrajectoryHits trajectory_points_;
    std::vector<EDEPTrajectory> children_trajectories_;
    std::map<component, bool> exiting_map_;
    std::map<component, bool> entering_map_;
    EDEPTrajectory* parent_trajectory_;
    int id_;
    int parent_id_;
    int pdg_code_;
    int depth_;

};





#endif