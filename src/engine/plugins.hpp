#pragma once

#include "engine/asset/Plugin.hpp"
#include "engine/logger/Plugin.hpp"
#include "engine/renderer/Plugin.hpp"
#include "engine/window/Plugin.hpp"

namespace engine::plugins {

using AssetManager = asset::Plugin;
using Logger       = logger::Plugin;
using Renderer     = renderer::Plugin;
using Window       = window::Plugin;

}   // namespace engine::plugins
