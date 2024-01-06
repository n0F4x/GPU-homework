#pragma once

#include <filesystem>

#include <tl/optional.hpp>

#include <fastgltf/types.hpp>

#include "engine/renderer/Allocator.hpp"

#include "Model.hpp"
#include "StagingModel.hpp"

namespace engine::scene {

class ModelFactory {
public:
    [[nodiscard]] static auto load_gltf(
        const std::filesystem::path& t_filepath,
        const renderer::Allocator&   t_allocator
    ) noexcept -> tl::optional<StagingModel>;
};

}   // namespace engine::scene
