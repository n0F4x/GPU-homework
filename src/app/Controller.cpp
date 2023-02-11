#include "Controller.hpp"

namespace app {

Controller::Controller(StateMachine& t_stateMachine) noexcept
    : m_stateMachine{ t_stateMachine } {}

auto Controller::running() const noexcept -> bool {
    return m_running;
}

void Controller::quit() noexcept {
    m_running = false;
}

auto Controller::stateMachine() noexcept -> StateMachine& {
    return m_stateMachine;
}

}   // namespace app