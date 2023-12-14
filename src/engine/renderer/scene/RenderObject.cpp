#include "RenderObject.hpp"

namespace engine::renderer {

[[nodiscard]] auto create_descriptor_set(
    vk::Device               t_device,
    vk::DescriptorSetLayout  t_descriptor_set_layout,
    vk::DescriptorPool       t_descriptor_pool,
    vk::DescriptorBufferInfo t_descriptor_buffer_info
) noexcept -> vk::DescriptorSet
{
    vk::DescriptorSetAllocateInfo descriptor_set_allocate_info{
        .descriptorPool     = t_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &t_descriptor_set_layout,
    };
    auto [result, descriptor_sets]{
        t_device.allocateDescriptorSets(descriptor_set_allocate_info)
    };
    if (result != vk::Result::eSuccess) {
        return nullptr;
    }

    vk::WriteDescriptorSet write_descriptor_set{
        .dstSet          = descriptor_sets.front(),
        .dstBinding      = 0,
        .descriptorCount = 1u,
        .descriptorType  = vk::DescriptorType::eUniformBuffer,
        .pBufferInfo     = &t_descriptor_buffer_info,
    };

    t_device.updateDescriptorSets(1, &write_descriptor_set, 0, nullptr);

    return descriptor_sets.front();
}

auto RenderObject::Mesh::create(
    const Device&           t_device,
    vk::DescriptorSetLayout t_descriptor_set_layout,
    vk::DescriptorPool      t_descriptor_pool,
    std::vector<Primitive>  t_primitives,
    const UniformBlock&     t_uniform_block
) noexcept -> tl::optional<Mesh>
{
    const vk::BufferCreateInfo buffer_create_info = {
        .size  = sizeof(t_uniform_block),
        .usage = vk::BufferUsageFlagBits::eUniformBuffer
               | vk::BufferUsageFlagBits::eTransferDst
    };
    const VmaAllocationCreateInfo allocation_create_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
               | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    return t_device
        .create_buffer(
            buffer_create_info, allocation_create_info, &t_uniform_block
        )
        .and_then(
            [&](std::tuple<vulkan::vma::Buffer, VmaAllocationInfo>&& result
            ) -> tl::optional<Mesh> {
                auto [_uniform_buffer, allocation_info]{ std::move(result) };
                auto buffer{ *_uniform_buffer };
                auto raw_descriptor_set{ create_descriptor_set(
                    *t_device,
                    t_descriptor_set_layout,
                    t_descriptor_pool,
                    vk::DescriptorBufferInfo{ .buffer = buffer,
                                              .offset = 0,
                                              .range  = sizeof(uniform_block) }
                ) };
                if (!raw_descriptor_set) {
                    return tl::nullopt;
                }
                return Mesh{
                    .primitives     = std::move(t_primitives),
                    .uniform_buffer = std::move(_uniform_buffer),
                    .descriptor_set = raw_descriptor_set,
                    .mapped         = allocation_info.pMappedData,
                    .uniform_block  = t_uniform_block,
                };
            }
        );
}

[[nodiscard]] static auto create_node(
    const Device&           t_device,
    vk::DescriptorSetLayout t_descriptor_set_layout,
    vk::DescriptorPool      t_descriptor_pool,
    const gfx::Model::Node& t_node,
    RenderObject::Node*     t_parent
) noexcept -> tl::optional<RenderObject::Node>
{
    tl::optional<RenderObject::Mesh> mesh;
    if (auto src_mesh = t_node.mesh; src_mesh.has_value()) {
        mesh = { RenderObject::Mesh::create(
            t_device,
            t_descriptor_set_layout,
            t_descriptor_pool,
            src_mesh->primitives,
            src_mesh->uniform_block
        ) };
        if (!mesh) {
            return tl::nullopt;
        }
    }

    RenderObject::Node node{
        .parent = t_parent,
        .mesh   = std::move(mesh),
        .matrix = t_node.matrix,
    };

    node.children.reserve(t_node.children.size());
    for (const auto& src_node : t_node.children) {
        auto child = create_node(
            t_device,
            t_descriptor_set_layout,
            t_descriptor_pool,
            src_node,
            &node
        );
        if (!child) {
            return tl::nullopt;
        }
        node.children.push_back(std::move(*child));
    }

    return node;
}

auto RenderObject::create(
    const Device&           t_device,
    vk::DescriptorSetLayout t_descriptor_set_layout,
    vk::DescriptorPool      t_descriptor_pool,
    const gfx::Model&       t_model,
    renderer::MeshBuffer&   t_mesh_buffer
) noexcept -> tl::optional<RenderObject>
{
    std::vector<Node> nodes;
    nodes.reserve(t_model.nodes().size());
    for (const auto& src_node : t_model.nodes()) {
        auto node = create_node(
            t_device,
            t_descriptor_set_layout,
            t_descriptor_pool,
            src_node,
            nullptr
        );
        if (!node) {
            return tl::nullopt;
        }
        nodes.push_back(std::move(*node));
    }

    return RenderObject{ std::move(nodes), t_mesh_buffer };
}

static auto draw_node(
    vk::CommandBuffer         t_graphics_buffer,
    vk::PipelineLayout        t_pipeline_layout,
    const RenderObject::Node& t_node
) noexcept -> void
{
    if (t_node.mesh) {
        auto& mesh{ t_node.mesh.value() };
        auto  matrix{ t_node.matrix };
        for (auto parent{ t_node.parent }; parent; parent = parent->parent) {
            matrix = parent->matrix * matrix;
        }
        memcpy(mesh.mapped, &matrix, sizeof(glm::mat4));

        for (const auto& primitive : t_node.mesh->primitives) {
            t_graphics_buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                t_pipeline_layout,
                0,
                t_node.mesh->descriptor_set,
                nullptr
            );

            if (primitive.index_count > 0) {
                t_graphics_buffer.drawIndexed(
                    primitive.index_count, 1, primitive.first_index_index, 0, 0
                );
            }
            else {
                t_graphics_buffer.draw(primitive.vertex_count, 1, 0, 0);
            }
        }
    }

    for (const auto& child : t_node.children) {
        draw_node(t_graphics_buffer, t_pipeline_layout, child);
    }
}

auto RenderObject::draw(
    vk::CommandBuffer  t_graphics_buffer,
    vk::PipelineLayout t_pipeline_layout
) const noexcept -> void
{
    m_mesh_buffer.bind(t_graphics_buffer);

    for (const auto& node : m_nodes) {
        draw_node(t_graphics_buffer, t_pipeline_layout, node);
    }
}

RenderObject::RenderObject(
    std::vector<Node>&&   t_nodes,
    renderer::MeshBuffer& t_mesh
) noexcept
    : m_nodes{ std::move(t_nodes) },
      m_mesh_buffer{ t_mesh }
{}

}   // namespace engine::renderer
