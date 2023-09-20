#pragma once

#include "App.hpp"

namespace engine {

template <typename Plugin>
concept PluginConcept = requires(Plugin t_plugin, App::Context& t_context) {
    {
        t_plugin.setup(t_context)
    };
};

}   // namespace engine
