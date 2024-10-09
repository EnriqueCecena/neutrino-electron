#include "EDEPHit.h"

std::map<component, std::string> component_to_string = {
  { component::GRAIN,  "LArHit"   },
  { component::STRAW,  "Straw"    },
  { component::ECAL,   "EMCalSci" },
  { component::MAGNET, "Magnet"   },
  { component::WORLD,  "World"    },
};

std::map<std::string, component> string_to_component = {
  { "LArHit",   component::GRAIN  },
  { "Straw",    component::STRAW  },
  { "EMCalSci", component::ECAL   },
  { "Magnet",   component::MAGNET },
  { "World",    component::WORLD  },
};

std::initializer_list<std::string> grain_names   = {"GRAIN", "GRIAN"};
std::initializer_list<std::string> stt_names     = {"horizontalST", "STT", "TrkMod", "CMod", "C3H6Mod"};
std::initializer_list<std::string> ecal_names    = {"ECAL", "kloe_calo_volume", "sand_inner"};
std::initializer_list<std::string> magnet_names  = {"KLOE", "Yoke", "Mag"};
std::initializer_list<std::string> world_names   = {"World", "rock", "volSAND"};
