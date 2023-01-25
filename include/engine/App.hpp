#pragma once

#include <vector>

#include <gsl/pointers>

#include "common/patterns/builder/helper.hpp"
#include "config/config.hpp"
#include "framework/SceneGraph.hpp"
#include "framework/StateMachine.hpp"
#include "Scheduler.hpp"

namespace engine {

class Controller;
class Stage;

class App final {
    ///----------------///
    ///  Member types  ///
    ///----------------///
    class Builder;
    friend BuilderBase<App>;

public:
    ///------------------------------///
    ///  Constructors / Destructors  ///
    ///------------------------------///
    [[nodiscard]] App(const App&) = delete;
    [[nodiscard]] App(App&&) noexcept = default;

    ///--------------------///
    ///  Member functions  ///
    ///--------------------///
    void run();

    ///------------------///
    ///  Static helpers  ///
    ///------------------///
    [[nodiscard]] static auto create() noexcept -> Builder;

private:
    [[nodiscard]] App() noexcept = default;

    ///--------------------///
    ///  Member variables  ///
    ///--------------------///
    std::string name = "App";
    fw::StateMachine stateMachine;
    fw::SceneGraph sceneGraph;
    Scheduler scheduler{ [this] { return sceneGraph.make_scene(); } };
};

class App::Builder final : public BuilderBase<App> {
public:
    ///------------------------------///
    ///  Constructors / Destructors  ///
    ///------------------------------///
    using BuilderBase<App>::BuilderBase;


    ///--------------------///
    ///  Member functions  ///
    ///--------------------///
    [[nodiscard]] auto set_name(std::string_view new_name) noexcept -> Self;
    [[nodiscard]] auto add_state(fw::State&& state) -> Self;
    [[nodiscard]] auto add_stage(Stage&& stage) -> Self;
};

}   // namespace engine
