#pragma once

#include <vector>

#include <gsl/pointers>

#include "common/patterns/builder/helper.hpp"
#include "config/config.hpp"
#include "framework/SceneGraph.hpp"
#include "framework/StateMachine.hpp"
#include "Schedule.hpp"

namespace engine {

class Controller;
class Stage;

class App final {
public:
    ///----------------///
    ///  Member types  ///
    ///----------------///
    class Builder;

    ///-----------///
    ///  Friends  ///
    ///-----------///
    friend Schedule;

    ///--------------------///
    ///  Member functions  ///
    ///--------------------///
    void run();

private:
    ///--------------------///
    ///  Member variables  ///
    ///--------------------///
    std::string m_name = "App";
    fw::fsm::StateMachine m_stateMachine;
    fw::SceneGraph m_sceneGraph;
    Schedule m_schedule{ *this };
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
    [[nodiscard]] auto set_name(std::string_view t_name) noexcept -> Builder&;
    [[nodiscard]] auto add_state(fw::State&& t_state,
                                 bool t_setAsInitialState = false) -> Builder&;
    [[nodiscard]] auto add_stage(Stage&& t_stage) -> Builder&;
};

}   // namespace engine
