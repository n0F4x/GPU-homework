#pragma once

#include <utility>

#include <gsl/pointers>

#include "config/id.hpp"
#include "patterns/builder/helper.hpp"

class App;


class State final {
  ///----------------///
 ///  Member types  ///
///----------------///
public:
    using Action = gsl::not_null<void(*)()>;

private:
    class Builder;
    friend BuilderBase<State>;

    friend App;

    constexpr static void empty_action() noexcept { /*empty by default*/ }

public:
  ///------------------------------///
 ///  Constructors / Destructors  ///
///------------------------------///
    constexpr [[nodiscard]] State(const State&) = delete;
    constexpr [[nodiscard]] State(State&&) noexcept = default;

  ///--------------------///
 ///  Member functions  ///
///--------------------///
    constexpr [[nodiscard]] auto get_id() const noexcept;
    constexpr void entered() const noexcept;
    constexpr void exited() const noexcept;

  ///------------------///
 ///  Static helpers  ///
///------------------///
    template<Id id>
        requires(id != 0)
    constexpr static [[nodiscard]] auto create() noexcept;
    constexpr static [[nodiscard]] gsl::not_null<const State*> invalid_state() noexcept;
    constexpr static [[nodiscard]] auto invalid(const State& state) noexcept;

private:
    constexpr explicit [[nodiscard]] State(Id id = {}) noexcept : id{ id } {}

  ///--------------------///
 ///  Member variables  ///
///--------------------///
    static const State s_invalid_state;

    Id id{};
    Action onEnter = empty_action;
    Action onExit = empty_action;
};


class State::Builder final : public BuilderBase<State> {
public:
  ///------------------------------///
 ///  Constructors / Destructors  ///
///------------------------------///
    using BuilderBase<State>::BuilderBase;

  ///--------------------///
 ///  Member functions  ///
///--------------------///
    constexpr [[nodiscard]] auto on_enter(Action callback) noexcept;
    constexpr [[nodiscard]] auto on_exit(Action callback) noexcept;
};


/// ////////////////////// ///
///     IMPLEMENTATION     ///
/// ////////////////////// ///

constexpr inline State State::s_invalid_state;


template<Id id>
    requires(id != 0)
constexpr [[nodiscard]] auto State::create() noexcept {
    return Builder{ id };
}

constexpr [[nodiscard]] gsl::not_null<const State*> State::invalid_state() noexcept {
    return &s_invalid_state;
}

constexpr [[nodiscard]] auto State::invalid(const State& state) noexcept {
    return state.id == 0;
}


constexpr [[nodiscard]] auto State::get_id() const noexcept {
    return id;
}

constexpr void State::entered() const noexcept {
    onEnter();
}

constexpr void State::exited() const noexcept {
    onExit();
}


constexpr [[nodiscard]] auto State::Builder::on_enter(Action callback) noexcept {
    draft().onEnter = callback;

    return std::move(*this);
}

constexpr [[nodiscard]] auto State::Builder::on_exit(Action callback) noexcept {
    draft().onExit = callback;

    return std::move(*this);
}
