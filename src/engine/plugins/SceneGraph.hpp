#pragma once

#include "engine/app/Plugin.hpp"

namespace engine::plugins {

class SceneGraph {
public:
    ///-----------///
    ///  Methods  ///
    ///-----------///
    auto inject(App::Context& t_context) const noexcept -> void;
};

static_assert(PluginConcept<SceneGraph>);

}   // namespace engine::plugins
