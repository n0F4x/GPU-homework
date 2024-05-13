#pragma once

#include <future>

#include "core/graphics/model/Model.hpp"
#include "core/renderer/base/allocator/Allocator.hpp"
#include "core/renderer/base/descriptor_pool/DescriptorPool.hpp"
#include "core/renderer/material_system/Effect.hpp"

namespace core::renderer {

class RenderModel {
public:
    class Requirements;

    struct DescriptorSetLayoutCreateInfo {
        uint32_t max_image_count;
        uint32_t max_sampler_count;
    };

    struct PipelineCreateInfo {
        Effect             effect;
        vk::PipelineLayout layout;
        vk::RenderPass     render_pass;
    };

    [[nodiscard]]
    static auto descriptor_pool_sizes() -> std::vector<vk::DescriptorPoolSize>;

    [[nodiscard]]
    static auto create_loader(
        vk::Device                                  t_device,
        const Allocator&                            t_allocator,
        std::span<const vk::DescriptorSetLayout, 3> t_descriptor_set_layouts,
        const PipelineCreateInfo&                   t_pipeline_create_info,
        vk::DescriptorPool                          t_descriptor_pool,
        cache::Handle<graphics::Model>              t_model
    ) -> std::packaged_task<RenderModel(vk::CommandBuffer)>;

    [[nodiscard]]
    static auto create_descriptor_set_layouts(
        vk::Device                           t_device,
        const DescriptorSetLayoutCreateInfo& info
    ) -> std::array<vk::UniqueDescriptorSetLayout, 3>;
    [[nodiscard]]
    static auto push_constant_range() noexcept -> vk::PushConstantRange;

    auto draw(
        vk::CommandBuffer  t_graphics_command_buffer,
        vk::PipelineLayout t_pipeline_layout
    ) const noexcept -> void;

private:
    Buffer m_index_buffer;

    // Base descriptor set
    Buffer            m_vertex_buffer;
    vk::DeviceAddress m_vertex_buffer_address;
    MappedBuffer      m_vertex_uniform;

    Buffer            m_transform_buffer;
    vk::DeviceAddress m_transform_buffer_address;
    MappedBuffer      m_transform_uniform;

    vk::UniqueDescriptorSet m_base_descriptor_set;

    // Image descriptor set
    std::vector<Image>               m_images;
    std::vector<vk::UniqueImageView> m_image_views;
    vk::UniqueDescriptorSet          m_image_descriptor_set;

    // Sampler descriptor set
    std::vector<vk::UniqueSampler> m_samplers;
    vk::UniqueDescriptorSet        m_sampler_descriptor_set;

    // Pipelines
    vk::UniquePipeline m_pipeline;

    cache::Handle<graphics::Model> m_model;


    explicit RenderModel(
        vk::Device                         t_device,
        Buffer&&                           t_index_buffer,
        Buffer&&                           t_vertex_buffer,
        MappedBuffer&&                     t_vertex_uniform,
        Buffer&&                           t_transform_buffer,
        MappedBuffer&&                     t_transform_uniform,
        vk::UniqueDescriptorSet&&          t_base_descriptor_set,
        std::vector<Image>&&               t_images,
        std::vector<vk::UniqueImageView>&& t_image_views,
        vk::UniqueDescriptorSet&&          t_image_descriptor_set,
        std::vector<vk::UniqueSampler>&&   t_samplers,
        vk::UniqueDescriptorSet&&          t_sampler_descriptor_set,
        vk::UniquePipeline&&               t_pipeline,
        cache::Handle<graphics::Model>&&   t_model
    );
};

}   // namespace core::renderer
