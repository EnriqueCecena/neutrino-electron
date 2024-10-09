#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <initializer_list>

#include "TG4Event.h"

enum class component {
  GRAIN,
  STRAW,
  ECAL,
  MAGNET,
  WORLD
};

extern std::map<component, std::string> component_to_string;

extern std::map<std::string, component> string_to_component;

extern std::initializer_list<std::string> grain_names;
extern std::initializer_list<std::string> stt_names;
extern std::initializer_list<std::string> ecal_names;
extern std::initializer_list<std::string> magnet_names;
extern std::initializer_list<std::string> world_names;

#endif