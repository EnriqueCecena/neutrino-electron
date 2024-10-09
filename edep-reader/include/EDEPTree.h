#include <iostream>

#include "EDEPTrajectory.h"

// Binary search not implemented due to low counts of child elements

class EDEPTree : public EDEPTrajectory {
    
  public:
    // Iterators
    typedef std::vector<EDEPTrajectory>::iterator child_iterator;
    class iterator {

      public:
        typedef EDEPTrajectory value_type;
        typedef std::forward_iterator_tag iterator_category;
        typedef ptrdiff_t difference_type;
        typedef value_type* pointer;
        typedef value_type& reference;

        reference operator *  () {return *current_it_;};
        pointer operator -> () {return &(*current_it_);};
        bool operator == (const iterator& it) {return (this->parent_trj_ == it.parent_trj_ && this->current_it_ == it.current_it_);};
        bool operator != (const iterator& it) {return (this->parent_trj_ != it.parent_trj_ || this->current_it_ != it.current_it_);};
        iterator& operator ++ (); //++it
        iterator  operator ++ (int) {iterator tmpIt = *this; ++*this; return tmpIt;}; //it++
        // iterator& operator -- (); //--it
        // iterator  operator -- (int); //it--

      private:
        iterator(pointer parent_trj_, child_iterator child_it);

        pointer parent_trj_ = nullptr;
        child_iterator current_it_;

      friend class EDEPTree;
    };
    
    typedef std::vector<EDEPTrajectory>::const_iterator const_child_iterator;
    class const_iterator {

      public:
        typedef const EDEPTrajectory value_type;
        typedef std::forward_iterator_tag iterator_category;
        typedef ptrdiff_t difference_type;
        typedef value_type* pointer;
        typedef value_type& reference;

        reference operator *  () {return *current_it_;};
        pointer operator -> () {return &(*current_it_);};
        bool operator == (const const_iterator& it) {return (this->parent_trj_ == it.parent_trj_ && this->current_it_ == it.current_it_);};
        bool operator != (const const_iterator& it) {return (this->parent_trj_ != it.parent_trj_ || this->current_it_ != it.current_it_);};
        const_iterator& operator ++ (); //++it
        const_iterator  operator ++ (int) {const_iterator tmpIt = *this; ++*this; return tmpIt;}; //it++
        // const_iterator& operator -- (); //--it
        // const_iterator  operator -- (int); //it--

      private:
        const_iterator(pointer parent_trj_, const_child_iterator child_it);

        pointer parent_trj_ = nullptr;
        const_child_iterator current_it_;

      friend class EDEPTree;
    };

    // Constructor
    EDEPTree();

    // Functions
          iterator begin()       {return iterator(nullptr, this->GetChildrenTrajectories().begin());}
    const_iterator begin() const {return const_iterator(nullptr, this->GetChildrenTrajectories().begin());}
          iterator end()         {return iterator(nullptr, this->GetChildrenTrajectories().end());}
    const_iterator end()   const {return const_iterator(nullptr, this->GetChildrenTrajectories().end());}


    void InizializeFromEdep(const TG4Event& edep_event, TGeoManager* geo);
    void InizializeFromTrj(const std::vector<EDEPTrajectory>& trajectories_vect);
        
    void AddTrajectory  (const EDEPTrajectory& trajectory);
    void AddTrajectoryTo(const EDEPTrajectory& trajectory, iterator it);
    
    void RemoveTrajectory    (int trj_id);
    void RemoveTrajectoryFrom(int trj_id, iterator it);
    void MoveTrajectoryTo    (int id_to_move, int next_parent_id);

    bool HasTrajectory (int trj_id) const;
    bool IsTrajectoryIn(int trj_id, iterator it);
    bool IsTrajectoryIn(int trj_id, const_iterator it) const;
    
          iterator GetTrajectory(int trj_id)       {return std::find_if(this->begin(), this->end(), [trj_id](const EDEPTrajectory& trj){ return trj_id == trj.GetId();});}
    const_iterator GetTrajectory(int trj_id) const {return std::find_if(this->begin(), this->end(), [trj_id](const EDEPTrajectory& trj){ return trj_id == trj.GetId();});}
          
          iterator GetParentOf(int trj_id);
    const_iterator GetParentOf(int trj_id) const;
          iterator GetParentOf(int trj_id, iterator it);
    const_iterator GetParentOf(int trj_id, const_iterator it) const;

          iterator GetTrajectoryFrom(int trj_id, iterator it);
    const_iterator GetTrajectoryFrom(int trj_id, const_iterator it) const;

          iterator GetTrajectoryEnd(iterator start);
    const_iterator GetTrajectoryEnd(const_iterator start) const;

          iterator  GetTrajectoryWithHitIdInDetector(int id, component component_name);
    const_iterator  GetTrajectoryWithHitIdInDetector(int id, component component_name) const;

    template <typename OutputIterator, typename F>
    OutputIterator Filter(OutputIterator out_it, F&& funct) {
            for (auto first = this->begin(); first != this->end(); ++first)
            {
                if (std::forward<F>(funct)(*first))
                {
                    *out_it = *first;
                    ++out_it;
                }
            }
        
            return out_it;
        
    }

  private:  
    void CreateTree(const std::vector<EDEPTrajectory>& trajectories_vect);
};

