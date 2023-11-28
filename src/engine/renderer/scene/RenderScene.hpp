#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <tl/optional.hpp>

#include "engine/graphics/Model.hpp"
#include "engine/renderer/Device.hpp"

#include "MeshBuffer.hpp"
#include "RenderObject.hpp"

namespace engine::renderer {

class ModelHandle;

class RenderScene {
public:
    ///-----------///
    ///  Methods  ///
    ///-----------///
    [[nodiscard]] auto load(const Device& t_device, gfx::Model t_model) noexcept
        -> tl::optional<ModelHandle>;

    auto flush(vk::CommandBuffer t_copy_buffer) noexcept -> void;

private:
    ///*************///
    ///  Variables  ///
    ///*************///
    std::vector<std::unique_ptr<MeshBuffer>> m_mesh_buffers;
    std::vector<StagingMeshBuffer>           m_staging_mesh_buffers;

    std::vector<std::unique_ptr<gfx::Model>> m_models;
};

class ModelHandle {
public:
    ///-----------///
    ///  Methods  ///
    ///-----------///
    [[nodiscard]] auto spawn(const Device& t_device) const noexcept
        -> tl::optional<RenderObject>;

private:
    ///******************///
    ///  Friend Classes  ///
    ///******************///
    friend RenderScene;

    ///*************///
    ///  Variables  ///
    ///*************///
    MeshBuffer& m_mesh_buffer;
    gfx::Model& m_model;

    ///******************************///
    ///  Constructors / Destructors  ///
    ///******************************///
    explicit ModelHandle(
        MeshBuffer& t_mesh_buffer,
        gfx::Model& t_model
    ) noexcept;
};

}   // namespace engine::renderer