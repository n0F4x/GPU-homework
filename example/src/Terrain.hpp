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

    [[nodiscard]]
    auto vertex_uniform() const noexcept -> const core::renderer::MappedBuffer&;

    [[nodiscard]]
    auto heightmap_image_view() const noexcept -> const vk::UniqueImageView&;

    [[nodiscard]]
    auto heightmap_sampler() const noexcept -> const vk::UniqueSampler&;

    auto draw(vk::CommandBuffer graphics_command_buffer) const -> void;

private:
    core::renderer::Buffer       m_vertex_buffer;
    core::renderer::MappedBuffer m_vertex_uniform;
    glm::u32vec2                 m_quad_count;
    core::renderer::Image        m_heightmap;
    vk::UniqueImageView          m_heightmap_view;
    vk::UniqueSampler            m_heightmap_sampler;

    explicit Terrain(
        vk::Device                     device,
        core::renderer::Buffer&&       vertex_buffer,
        core::renderer::MappedBuffer&& vertex_uniform,
        glm::u32vec2                   quad_count,
        core::renderer::Image&&        heightmap,
        vk::UniqueImageView&&          heightmap_view,
        vk::UniqueSampler&&            heightmap_sampler
    ) noexcept;
};
