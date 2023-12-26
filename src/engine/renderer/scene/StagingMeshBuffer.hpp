#pragma once

#include <tl/optional.hpp>

#include <vulkan/vulkan.hpp>

#include "engine/renderer/Device.hpp"
#include "engine/utility/vma/Buffer.hpp"

#include "MeshBuffer.hpp"

namespace engine::renderer {

class MeshBuffer;

class StagingMeshBuffer {
public:
    ///----------------///
    /// Static methods ///
    ///----------------///
    template <typename Vertex>
    [[nodiscard]] static auto create(
        const Device&             t_device,
        std::span<const Vertex>   t_vertices,
        std::span<const uint32_t> t_indices
    ) noexcept -> tl::optional<StagingMeshBuffer>;

    ///-----------///
    ///  Methods  ///
    ///-----------///
    [[nodiscard]] auto upload(
        const Device&     t_device,
        vk::CommandBuffer t_copy_command_buffer
    ) const noexcept -> tl::optional<MeshBuffer>;

private:
    ///*************///
    ///  Variables  ///
    ///*************///
    vulkan::vma::Buffer m_vertex_staging_buffer;
    vulkan::vma::Buffer m_index_staging_buffer;
    uint32_t            m_vertex_buffer_size;
    uint32_t            m_index_buffer_size;
    uint32_t            m_index_count;

    ///******************************///
    ///  Constructors / Destructors  ///
    ///******************************///
    explicit StagingMeshBuffer(
        vulkan::vma::Buffer&& t_vertex_staging_buffer,
        vulkan::vma::Buffer&& t_index_staging_buffer,
        uint32_t              t_vertex_buffer_size,
        uint32_t              t_index_buffer_size,
        uint32_t              t_index_count
    ) noexcept;
};

}   // namespace engine::renderer

#include "StagingMeshBuffer.inl"