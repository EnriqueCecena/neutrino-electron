#include "EDEPTree.h"

// Iterators
EDEPTree::iterator::iterator(value_type* parent_trj, child_iterator child_it) {
  this->current_it_ = child_it;
  this->parent_trj_ = parent_trj;
}

EDEPTree::iterator& EDEPTree::iterator::operator ++ () {
  if(!current_it_->GetChildrenTrajectories().empty()) {
    parent_trj_ = &(*current_it_);
    current_it_ = current_it_->GetChildrenTrajectories().begin();

  } else {
    ++current_it_;
    while(parent_trj_ && (current_it_ == parent_trj_->GetChildrenTrajectories().end())) {
      auto grand_parent = parent_trj_->GetParent();
      if(!grand_parent) {
        current_it_ = parent_trj_->GetChildrenTrajectories().end();
        parent_trj_ = nullptr;
        break;
      }

      auto b = grand_parent->GetChildrenTrajectories().begin();

      // syntax for doing parent_trj_ + 1
      ptrdiff_t diff = parent_trj_  - &(*b); // granted by std::vector

      parent_trj_ = grand_parent;
      current_it_ = b + diff + 1;
    }
  }

  return *this;
}

EDEPTree::const_iterator::const_iterator(value_type* parent_trj, const_child_iterator child_it) {
  this->current_it_ = child_it;
  this->parent_trj_ = parent_trj;
}

EDEPTree::const_iterator& EDEPTree::const_iterator::operator ++ () {
  if(!current_it_->GetChildrenTrajectories().empty()) {
    parent_trj_ = &(*current_it_);
    current_it_ = current_it_->GetChildrenTrajectories().begin();

  } else {
    ++current_it_;
    while(parent_trj_ && (current_it_ == parent_trj_->GetChildrenTrajectories().end())) {
      auto grand_parent = parent_trj_->GetParent();
      if(!grand_parent) {
        current_it_ = parent_trj_->GetChildrenTrajectories().end();
        parent_trj_ = nullptr;
        break;
      }

      auto b = grand_parent->GetChildrenTrajectories().begin();

      // syntax for doing parent_trj_ + 1
      ptrdiff_t diff = parent_trj_  - &(*b); // granted by std::vector

      parent_trj_ = grand_parent;
      current_it_ = b + diff + 1;
    }
  }

  return *this;
}

EDEPTree::EDEPTree() {
  this->SetId(-1);
  this->SetParent(nullptr);
  this->SetDepth(-1);

}

void EDEPTree::CreateTree(const std::vector<EDEPTrajectory>& trajectories_vect) {
  bool complete = true;
  while (complete) {
    complete = true;
    for (auto it = trajectories_vect.begin(); it != trajectories_vect.end(); it++) {

      if (!HasTrajectory((*it).GetId())) {
        complete = false;
        int depth = 0;
        AddTrajectory((*it));
      }
    }
  }
  std::for_each(this->begin(), this->end(), [] (EDEPTrajectory& trj) {trj.ComputeDepth();});
}

void EDEPTree::InizializeFromEdep(const TG4Event& edep_event, TGeoManager* geo) {
  this->GetChildrenTrajectories().clear();
  std::vector<EDEPTrajectory> trajectories;
  for (auto trj:edep_event.Trajectories) {
    trajectories.push_back(EDEPTrajectory(trj, edep_event.SegmentDetectors, geo)); 
  }
  CreateTree(trajectories);
}

void EDEPTree::InizializeFromTrj(const std::vector<EDEPTrajectory>& trajectories_vect) {
  this->GetChildrenTrajectories().clear();
  CreateTree(trajectories_vect);
}

void EDEPTree::AddTrajectory(const EDEPTrajectory& trajectory) {
  int parent_id = trajectory.GetParentId();
  if (parent_id == -1) {
    this->AddChild(trajectory);
  } else {
    std::find_if(this->begin(), this->end(), 
              [parent_id](const EDEPTrajectory& trj){ return parent_id == trj.GetId();})->AddChild(trajectory);
  }
}

void EDEPTree::AddTrajectoryTo(const EDEPTrajectory& trajectory, iterator it) {
  int parent_id = trajectory.GetParentId();
  if (parent_id == -1) {
    this->AddChild(trajectory);
  } else {
    EDEPTree::iterator end_it = GetTrajectoryEnd(it);
    std::find_if(it, end_it, 
              [parent_id](const EDEPTrajectory& trj){ return parent_id == trj.GetId();})->AddChild(trajectory);
  }
}

void EDEPTree::RemoveTrajectory(int trj_id) {
  GetParentOf(trj_id)->RemoveChildWithId(trj_id);
}

void EDEPTree::RemoveTrajectoryFrom(int trj_id, iterator it) {
  GetParentOf(trj_id, it)->RemoveChildWithId(trj_id);
}

void EDEPTree::MoveTrajectoryTo(int id_to_move, int next_parent_id) {
  EDEPTrajectory trjToMove = *GetTrajectory(id_to_move);
  RemoveTrajectory(id_to_move);
  GetTrajectory(next_parent_id)->AddChild(trjToMove);
  std::for_each(this->begin(), this->end(), [] (EDEPTrajectory& trj) {trj.ComputeDepth();});
}

bool EDEPTree::HasTrajectory(int trj_id) const {
  return std::find_if(this->begin(), this->end(), 
                      [trj_id](const EDEPTrajectory& trj){ return trj_id == trj.GetId();}) != this->end();
}

bool EDEPTree::IsTrajectoryIn(int trj_id, iterator it) {
  EDEPTree::iterator end_it = GetTrajectoryEnd(it);
  EDEPTree::iterator found_it = std::find_if(it, end_it, [trj_id](const EDEPTrajectory& trj){ return trj_id == trj.GetId();});
  return (found_it != end_it) ? true : false;
}

bool EDEPTree::IsTrajectoryIn(int trj_id, const_iterator it) const {
  EDEPTree::const_iterator end_it = GetTrajectoryEnd(it);
  EDEPTree::const_iterator found_it = std::find_if(it, end_it, [trj_id](const EDEPTrajectory& trj){ return trj_id == trj.GetId();});
  return (found_it != end_it) ? true : false;
}

EDEPTree::iterator EDEPTree::GetParentOf(int trj_id) {
  auto f = [trj_id](const EDEPTrajectory& trj){ for (const auto& t:trj.GetChildrenTrajectories()) { if (trj_id == t.GetId()) return true;}; return false;};
  return std::find_if(this->begin(), this->end(), f);
}

EDEPTree::const_iterator EDEPTree::GetParentOf(int trj_id) const {
  auto f = [trj_id](const EDEPTrajectory& trj){ for (const auto& t:trj.GetChildrenTrajectories()) { if (trj_id == t.GetId()) return true;}; return false;};
  return std::find_if(this->begin(), this->end(), f);
}

EDEPTree::iterator EDEPTree::GetParentOf(int trj_id, iterator it) {
  EDEPTree::iterator end_it = GetTrajectoryEnd(it);
  auto f = [trj_id](const EDEPTrajectory& trj){ for (const auto& t:trj.GetChildrenTrajectories()) { if (trj_id == t.GetId()) return true;}; return false;};
  EDEPTree::iterator found_it = std::find_if(it, end_it, f);
  return (found_it != end_it) ? found_it : this->end();
}

EDEPTree::const_iterator EDEPTree::GetParentOf(int tid, const_iterator it) const {
  EDEPTree::const_iterator end_it = GetTrajectoryEnd(it);
  auto f = [tid](const EDEPTrajectory& trj){ for (const auto& t:trj.GetChildrenTrajectories()) { if (tid == t.GetId()) return true;}; return false;};
  EDEPTree::const_iterator found_it = std::find_if(it, end_it, f);
  return (found_it != end_it) ? found_it : this->end();
}

EDEPTree::iterator EDEPTree::GetTrajectoryFrom(int trj_id, iterator it) {
  EDEPTree::iterator end_it = GetTrajectoryEnd(it);
  EDEPTree::iterator found_it = std::find_if(it, end_it, [trj_id](const EDEPTrajectory& trj){ return trj_id == trj.GetId();});
  return (found_it != end_it) ? found_it : end_it;
}

EDEPTree::const_iterator EDEPTree::GetTrajectoryFrom(int trj_id, const_iterator it) const {
  EDEPTree::const_iterator end_it = GetTrajectoryEnd(it);
  EDEPTree::const_iterator found_it = std::find_if(it, end_it, [trj_id](const EDEPTrajectory& trj){ return trj_id == trj.GetId();});
  return (found_it != end_it) ? found_it : this->end();
}

EDEPTree::iterator  EDEPTree::GetTrajectoryWithHitIdInDetector(int id, component component_name) {
  EDEPTree::iterator found_it = std::find_if(this->begin(), this->end(), 
                      [id, component_name](const EDEPTrajectory& trj){ return trj.HasHitWithIdInDetector(id, component_name);});
  return (found_it != this->end()) ? found_it : this->end();
}

EDEPTree::const_iterator  EDEPTree::GetTrajectoryWithHitIdInDetector(int id, component component_name) const {
  EDEPTree::const_iterator found_it = std::find_if(this->begin(), this->end(), 
                      [id, component_name](const EDEPTrajectory& trj){ return trj.HasHitWithIdInDetector(id, component_name);});
  return (found_it != this->end()) ? found_it : this->end();
}

EDEPTree::iterator EDEPTree::GetTrajectoryEnd(iterator start) {
  EDEPTree::iterator end_it = start;
  if(!start->GetChildrenTrajectories().empty()) {
    end_it = ++iterator(start->Get(), --(start->GetChildrenTrajectories().end()));
  } else {
    end_it = ++start;
  }
  return end_it;
}

EDEPTree::const_iterator EDEPTree::GetTrajectoryEnd(const_iterator start) const {
  EDEPTree::const_iterator end_it = start;
  if(!start->GetChildrenTrajectories().empty()) {
    end_it = ++const_iterator(start->Get(), --(start->GetChildrenTrajectories().end()));
  } else {
    end_it = ++start;
  }
  return end_it;
}


