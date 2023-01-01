#pragma once

#include <vector>

#include <gsl/pointers>

#include "config/id.hpp"
#include "patterns/builder/helper.hpp"
#include "engine/StateMachine.hpp"

class Controller;
class StateMachine;
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
    static [[nodiscard]] auto create() noexcept -> Builder;

private:
    [[nodiscard]] App() noexcept = default;

  ///--------------------///
 ///  Member variables  ///
///--------------------///

    std::string name = "App";

    StateMachine stateMachine;

    std::vector<Stage> stages;
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
    [[nodiscard]] auto add_state(State&& state) -> Self;
    [[nodiscard]] auto add_stage(Stage&& stage) -> Self;
};
