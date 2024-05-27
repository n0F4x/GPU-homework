#pragma once

#include <future>

#include <glm/vec2.hpp>

#include <core/renderer/base/allocator/Allocator.hpp>
#include <core/renderer/memory/Buffer.hpp>
#include <core/renderer/memory/Image.hpp>

class Terrain {
public:
    struct Vertex {
        glm::vec2 position;
        glm::vec2 texture_coord;
    };

    [[nodiscard]]
    static auto
        create_loader(vk::Device device, const core::renderer::Allocator& allocator)
            -> std::packaged_task<Terrain(vk::CommandBuffer)>;

private:
    core::renderer::Buffer m_vertex_buffer;
    uint32_t               m_vertex_count{};
    core::renderer::Image  m_heightmap;
    vk::UniqueImageView    m_heightmap_view;
    vk::UniqueSampler      m_heightmap_sampler;

    explicit Terrain(
        core::renderer::Buffer&& vertex_buffer,
        uint32_t                 vertex_count,
        core::renderer::Image&&  heightmap,
        vk::UniqueImageView&&    heightmap_view,
        vk::UniqueSampler&&      heightmap_sampler
    ) noexcept;
};
