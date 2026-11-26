#pragma once
#include "log.h"
namespace jq {
inline IdentityVisibility validate_observation(const std::string& text, const Rules& rules) {
  auto o = parse_json(text);
  if (o.at("schema_version").integer() != 2 || !o.at("own_identity_visible").boolean() ||
      o.at("collision").str() != "result_only" || o.at("marshal_reveals_flag").boolean() != rules.marshal_reveals_flag)
    throw std::runtime_error("unsupported/inconsistent observation config");
  const auto ally = o.at("ally_visibility").str();
  const bool enemy = o.at("enemy_identity_visible").boolean();
  if (ally == "all") return enemy ? IdentityVisibility::AllIdentities : IdentityVisibility::AllyAndPublic;
  if (ally == "none" && !enemy) return IdentityVisibility::OwnAndPublic;
  throw std::runtime_error("unsupported identity visibility");
}

inline std::string match_observation_config(const std::string& text, const Rules& rules,
                                            IdentityVisibility visibility) {
  if (validate_observation(text,rules) != IdentityVisibility::OwnAndPublic)
    throw std::runtime_error("simulation starts from the base dark observation config");
  auto config=parse_json(text);
  if (visibility!=IdentityVisibility::OwnAndPublic) {
    const bool full=visibility==IdentityVisibility::AllIdentities;
    config.object["ally_visibility"]=parse_json("\"all\"");
    config.object["enemy_identity_visible"]=parse_json(full ? "true" : "false");
    config.object["id"]=parse_json(full ? "\"four-full-derived-1\"" : "\"four-dual-derived-1\"");
  }
  return dump_json(config);
}
}
