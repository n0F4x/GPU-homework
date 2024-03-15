#pragma once

#include <functional>
#include <vector>

#include <tl/optional.hpp>

#include "engine/common/Cache.hpp"
#include "engine/renderer/base/Allocator.hpp"
#include "engine/renderer/model/Model.hpp"

#include "Scene.hpp"

namespace engine::renderer {

class SceneBuilder {
public:
    explicit SceneBuilder(Cache& t_cache) noexcept;

    auto load_model(auto&&... t_args) -> tl::optional<Handle<Model>>;

    [[nodiscard]] auto build(const renderer::Allocator& t_allocator) && -> Scene;

private:
    std::reference_wrapper<Cache> m_cache;
    std::vector<Handle<Model>>    m_models;
};

}   // namespace engine::renderer

#include "SceneBuilder.inl"