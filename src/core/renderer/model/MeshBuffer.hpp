#pragma once

#include <vulkan/vulkan.hpp>

#include "core/renderer/memory/Buffer.hpp"

namespace core::renderer {

class StagingMeshBuffer;

class MeshBuffer {
public:
    ///------------------///
    ///  Nested classes  ///
    ///------------------///
    struct Vertices {
        Buffer buffer;
    };

    struct Indices {
        uint32_t count{};
        Buffer   buffer;
    };

    MeshBuffer(MeshBuffer&&) noexcept = default;

    ///-----------///
    ///  Methods  ///
    ///-----------///
    auto bind(vk::CommandBuffer t_command_buffer) const noexcept -> void;

private:
    ///******************///
    ///  Friend Classes  ///
    ///******************///
    friend StagingMeshBuffer;

    ///*************///
    ///  Variables  ///
    ///*************///
    Vertices m_vertices;
    Indices  m_indices;

    ///******************************///
    ///  Constructors / Destructors  ///
    ///******************************///
    explicit MeshBuffer(Vertices&& t_vertices, Indices&& t_indices) noexcept;
};

}   // namespace core::renderer
