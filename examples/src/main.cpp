#include <utility>

#include "engine/engine.hpp"

auto main() noexcept -> int
{
    using namespace engine;

    const auto result{
        App::create()
            .add_plugin<plugins::Logger>(plugins::Logger::Level::eTrace)
            .add_plugin<plugins::AssetManager>()
            .add_plugin<plugins::Window>(
                sf::VideoMode{ 450u, 600u },
                "My window",
                window::Window::Style::eDefault
            )
            .add_plugin<plugins::Renderer>()
            .add_plugin<plugins::SceneGraph>()
            .build_and_run([](App&) noexcept {})
    };

    return std::to_underlying(result);
}
