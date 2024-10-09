#include "EDEPTrajectory.h"
#include <math.h>  

TGeoNode* GetNode(const TG4TrajectoryPoint& tpoint, TGeoManager* geo) {
  TGeoNode* node = nullptr;

  TLorentzVector position = tpoint.GetPosition();
  TVector3       mom = tpoint.GetMomentum();

  node = geo->FindNode(position.X(), position.Y(), position.Z());
  
  geo->SetCurrentDirection(mom.X(), mom.Y(), mom.Z());
  geo->FindNextBoundary(1);

  if (geo->GetStep() < 1E-5) {
    geo->Step();
    node = geo->GetCurrentNode();
  }
  
  return node;
}

EDEPTrajectory::EDEPTrajectory(const TG4Trajectory& trajectory, const TG4HitSegmentDetectors& hits, TGeoManager* geo) :
                                               p0_(trajectory.GetInitialMomentum()),
                                               parent_trajectory_(nullptr),
                                               id_(trajectory.GetTrackId()), 
                                               parent_id_(trajectory.GetParentId()), 
                                               pdg_code_(trajectory.GetPDGCode()) {
      
  for (auto it = trajectory.Points.begin(); it != trajectory.Points.end(); ++it) {
    trajectory_points_.push_back(*it);

    auto next_it = std::next(it);

    if (next_it == trajectory.Points.end()) {
      continue;
    }

    auto current_node   = GetNode(*it, geo);
    auto current_volume = current_node->GetName();
    auto next_node   = GetNode(*next_it, geo);
    auto next_volume = next_node->GetName();

    bool in_grain   = Match(current_volume, grain_names);
    bool in_stt     = Match(current_volume, stt_names);
    bool in_ecal    = Match(current_volume, ecal_names);
    bool in_mag     = Match(current_volume, magnet_names);
    bool in_world   = Match(current_volume, world_names);

    bool next_grain = Match(next_volume,    grain_names);
    bool next_stt   = Match(next_volume,    stt_names);
    bool next_ecal  = Match(next_volume,    ecal_names);       
    bool next_mag   = Match(next_volume,    magnet_names);
    bool next_world = Match(next_volume,    world_names);

    if (!exiting_map_[component::GRAIN])   exiting_map_[component::GRAIN]   = (in_grain   && ( next_stt   || next_ecal  || next_mag   || next_world ));
    if (!exiting_map_[component::STRAW])   exiting_map_[component::STRAW]   = (in_stt     && ( next_grain || next_ecal  || next_mag   || next_world ));
    if (!exiting_map_[component::ECAL])    exiting_map_[component::ECAL]    = (in_ecal    && ( next_stt   || next_grain || next_mag   || next_world ));
    if (!exiting_map_[component::MAGNET])  exiting_map_[component::MAGNET]  = (in_mag     && ( next_stt   || next_ecal  || next_grain || next_world ));
    if (!exiting_map_[component::WORLD])   exiting_map_[component::WORLD]   = (in_world   && ( next_stt   || next_ecal  || next_mag   || next_grain ));

    if (!entering_map_[component::GRAIN])  entering_map_[component::GRAIN]  = (next_grain && ( in_stt     || in_ecal    || in_mag     || in_world   ));
    if (!entering_map_[component::STRAW])  entering_map_[component::STRAW]  = (next_stt   && ( in_grain   || in_ecal    || in_mag     || in_world   ));
    if (!entering_map_[component::ECAL])   entering_map_[component::ECAL]   = (next_ecal  && ( in_stt     || in_grain   || in_mag     || in_world   ));
    if (!entering_map_[component::MAGNET]) entering_map_[component::MAGNET] = (next_mag   && ( in_stt     || in_ecal    || in_grain   || in_world   ));
    if (!entering_map_[component::WORLD])  entering_map_[component::WORLD]  = (next_world && ( in_stt     || in_ecal    || in_mag     || in_grain   ));
  }

  for (const auto& hmap:hits) {
    std::vector<EDEPHit> edep_hits;
    for (uint i = 0; i < hmap.second.size(); i++) {
      auto h = hmap.second[i];
      if (id_ == h.Contrib[0]) {
        edep_hits.push_back(EDEPHit(h, i));
      }
    }
    if (!edep_hits.empty()) {
      std::sort(edep_hits.begin(), edep_hits.end(), 
                  [](EDEPHit i,EDEPHit j) { return i.GetStart().T() < j.GetStart().T();});
      hit_map_[string_to_component[hmap.first]] = edep_hits;
    }
  }
}

EDEPTrajectory::EDEPTrajectory(const EDEPTrajectory& trj) {
  *this = trj;
}

EDEPTrajectory& EDEPTrajectory::operator=(const EDEPTrajectory& trj) {
  this->id_ = trj.id_;
  this->parent_id_ = trj.parent_id_;
  this->parent_trajectory_ = trj.parent_trajectory_;
  this->pdg_code_ = trj.pdg_code_;
  this->p0_ = trj.p0_;
  this->depth_ = trj.depth_;
  this->children_trajectories_ = trj.children_trajectories_;
  for (auto& t:this->children_trajectories_) {
    t.parent_trajectory_ = this;
  }
  this->hit_map_ = trj.hit_map_;
  this->entering_map_ = trj.entering_map_;
  this->exiting_map_ = trj.exiting_map_;
  this->trajectory_points_ = trj.trajectory_points_;
  return *this;
}

bool EDEPTrajectory::operator==(const EDEPTrajectory& trj) { 
  return (
    this->id_ == trj.id_ && this->parent_id_ == trj.parent_id_ &&
    this->parent_trajectory_ == trj.parent_trajectory_ &&
    this->pdg_code_ == trj.pdg_code_ &&
    this->p0_ == trj.p0_ &&
    this->depth_ == trj.depth_ &&
    // this->children_trajectories_ == trj.children_trajectories_ &&
    // this->hit_map_ == trj.hit_map_ &&
    this->entering_map_ == trj.entering_map_ &&
    this->exiting_map_ == trj.exiting_map_ 
    // this->trajectory_points_ == trj.trajectory_points_
  );
}

EDEPTrajectory::EDEPTrajectory(EDEPTrajectory&& trj) {
  *this = std::move(trj);
}

EDEPTrajectory& EDEPTrajectory::operator=(EDEPTrajectory&& trj) {
  this->id_ = trj.id_;
  this->parent_id_ = trj.parent_id_;
  this->parent_trajectory_ = trj.parent_trajectory_;
  this->pdg_code_ = trj.pdg_code_;
  this->p0_ = trj.p0_;
  this->depth_ = trj.depth_;
  std::swap(this->children_trajectories_, trj.children_trajectories_);
  std::swap(this->hit_map_, trj.hit_map_);
  std::swap(this->entering_map_, trj.entering_map_);
  std::swap(this->exiting_map_, trj.exiting_map_);
  std::swap(this->trajectory_points_, trj.trajectory_points_);
  for (auto& t:this->children_trajectories_) {
    t.parent_trajectory_ = this;
  }

  trj.parent_trajectory_ = nullptr;
  return *this;
}

bool EDEPTrajectory::RemoveChildWithId(int child_id) {
  auto end = std::remove_if(children_trajectories_.begin(), children_trajectories_.end(), [child_id](const EDEPTrajectory& trj) {return trj.GetId() == child_id;});
  auto b = children_trajectories_.end() != end;
  children_trajectories_.erase(end, children_trajectories_.end());

  return b;
}

std::string EDEPTrajectory::Print(std::string& full_out, int depth, int current_depth) const {
  if (current_depth <= depth) {
    for ( int i = 0 ; i < current_depth ; i++ ) {
      if ( i != current_depth-1 ) {
        std::cout << "    ";
        full_out += "    ";
      }
      else {
        std::cout << "|-- ";
        full_out += "|-- ";
      }
    }
    
    std::cout <<  this->GetDepth() << " " << this->GetId() << " " << this << " " << this->GetPDGCode() << std::endl;
    full_out += std::to_string(this->GetDepth()) + " " + std::to_string(this->GetId()) + " " + std::to_string(this->GetPDGCode()) + "\n";

    for (auto el:hit_map_) {
      for ( int i = 0 ; i < current_depth ; i++ ) {
        if ( i != current_depth ) {
          std::cout << "    ";
          full_out += "    ";
        }
        else {
          std::cout << "|-- ";
          full_out += "|-- ";
        }
      }

      std::cout << component_to_string[el.first] << " " << el.second.size() << "; ";
      full_out += component_to_string[el.first] + " " + std::to_string(el.second.size()) + "; ";
      if (hit_map_.size() > 0) {
        std::cout << std::endl;
        full_out += "\n";
      }
    }
    for ( uint i = 0 ; i < this->children_trajectories_.size() ; ++i ) {
      this->children_trajectories_.at(i).Print( full_out, depth, current_depth+1);
    }
  }

  return full_out;
}

void EDEPTrajectory::ComputeDepth() {
  int depth = -1;
  auto tmp_parent = parent_trajectory_;
  while (tmp_parent != nullptr) {
    depth++;
    tmp_parent = tmp_parent->GetParent();
  }
  SetDepth(depth);
}


bool EDEPTrajectory::HasHitInDetector(component component_name) const {
  return hit_map_.find(component_name) != hit_map_.end() ;
}

bool EDEPTrajectory::HasHitBeforeTime(double time) const {
  return HasHitWhere([time](const EDEPHit& hit) {return hit.GetStart().T() < time;});
}

bool EDEPTrajectory::HasHitAfterTime(double time) const {
  return HasHitWhere([time](const EDEPHit& hit) {return hit.GetStart().T() > time;});
}

bool EDEPTrajectory::HasHitInTime(double start_time, double stop_time) const {
  return HasHitWhere([start_time, stop_time](const EDEPHit& hit) {return (hit.GetStart().T() > start_time && hit.GetStart().T() < stop_time);});
}

bool EDEPTrajectory::HasHitWithIdInDetector( int id, component component_name) const {
  if (GetHitMap().find(component_name) == GetHitMap().end()) return false;

  auto f = [id](const EDEPHit& hit) { return (id == hit.GetId()) ? true : false;};
  auto found_hit = std::find_if(GetHitMap().at(component_name).begin(), GetHitMap().at(component_name).end(), f);
  bool result = (found_hit != GetHitMap().at(component_name).end());
  return result;
}


double EDEPTrajectory::GetDepositedEnergy(component component_name) {
  double deposited_energy = 0;
  for (auto& hit:hit_map_[component_name]) {
    deposited_energy += hit.GetSecondaryDeposit();
  }
  return deposited_energy;
}

bool EDEPTrajectory::IsTrajectorySaturated() const {
  return (trajectory_points_.size() == 10000);
}

bool EDEPTrajectory::Match(std::string volume, std::initializer_list<std::string> names) const {
  for (const auto& n:names) {
    if (volume.find(n) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool EDEPTrajectory::IsEntering(component component_name) const {
  return entering_map_.at(component_name);
}

bool EDEPTrajectory::IsExiting(component component_name) const {
  return exiting_map_.at(component_name);
}
