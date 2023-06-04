#include "App.hpp"

namespace app {

//////////////////////////////
///----------------------- ///
///  App   IMPLEMENTATION  ///
///------------------------///
//////////////////////////////
App::App(Builder&& t_builder)
    : m_runner{ t_builder.runner() },
      m_window{ t_builder.window() } {}

void App::run() {
    m_runner(m_window);
}

auto App::create() noexcept -> App::Builder {
    return Builder{};
}

///////////////////////////////////////
///---------------------------------///
///  App::Builder   IMPLEMENTATION  ///
///---------------------------------///
///////////////////////////////////////
auto App::Builder::build() -> App {
    return App{ std::move(*this) };
}

auto App::Builder::set_runner(Runner&& t_runner) noexcept -> Builder& {
    m_runner = std::move(t_runner);
    return *this;
}

auto App::Builder::set_window(const Window::Builder& t_window_builder) noexcept
    -> App::Builder& {
    m_window_builder = t_window_builder;
    return *this;
}

auto App::Builder::runner() noexcept -> App::Runner {
    return std::move(m_runner);
}

auto App::Builder::window() noexcept -> const Window::Builder& {
    return m_window_builder;
}

}   // namespace app
