#pragma once

#include "engine/common/Plugin.hpp"

namespace engine::asset_manager {

class Plugin {
public:
    ///-------------///
    ///  Operators  ///
    ///-------------///
    auto operator()(Store& t_store) noexcept -> void;
};

static_assert(PluginConcept<Plugin>);

}   // namespace engine::asset_manager
