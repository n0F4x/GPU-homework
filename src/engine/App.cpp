#include "engine/App.hpp"

#include <iostream>
#include <format>
#include <algorithm>
#include <functional>

#include "engine/Controller.hpp"
#include "engine/State.hpp"
#include "engine/Stage.hpp"


void App::run() {
    *running = true;
    std::cout << std::format("{} is running...\n", name);

    currentState->entered();

    if (!std::ranges::empty(stages)) {
        Controller controller{ *this };

        while (*running) {
            std::ranges::for_each(stages, std::bind_back(&Stage::run, std::ref(controller)));

            transition();
        }
    }

    currentState->exited();
}


[[nodiscard]] auto App::create() noexcept -> Builder {
    return Builder{};
}


void App::transition() noexcept {
    if (nextState != currentState) {
        currentState->exited();

        prevState = currentState;
        currentState = nextState;

        currentState->entered();
    }
}


[[nodiscard]] auto App::Builder::set_name(std::string_view new_name) noexcept -> Self {
    draft().name = new_name;

    return std::move(*this);
}

[[nodiscard]] auto App::Builder::add_state(State&& state) -> Self {
    if (State::invalid(*draft().currentState))
        draft().currentState = &draft().states.try_emplace(state.get_id(), std::move(state)).first->second;
    else
        draft().states.try_emplace(state.get_id(), std::move(state));

    return std::move(*this);
}

[[nodiscard]] auto App::Builder::add_stage(Stage&& stage) -> Self {
    if (!Stage::empty(stage))
        draft().stages.push_back(std::move(stage));

    return std::move(*this);
}
