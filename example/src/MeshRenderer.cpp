#include "MeshRenderer.hpp"

#include <spdlog/spdlog.h>

#include <core/renderer/base/descriptor_pool/Builder.hpp>
#include <core/window/Window.hpp>

#include "demo_init.hpp"
#include "DependencyProvider.hpp"

using namespace core;

constexpr static uint32_t g_frame_count{ 1 };

[[nodiscard]]
static auto
    load_terrain(const renderer::Device& t_device, const renderer::Allocator& t_allocator)
        -> Terrain
{
    auto                                transfer_command_pool{ init::create_command_pool(
        t_device.get(), t_device.info().get_queue_index(vkb::QueueType::graphics).value()
    ) };
    const vk::CommandBufferAllocateInfo command_buffer_allocate_info{
        .commandPool        = transfer_command_pool.get(),
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    auto command_buffer{
        t_device->allocateCommandBuffers(command_buffer_allocate_info).front()
    };

    auto packaged_terrain{ Terrain::create_loader(t_device.get(), t_allocator) };

    constexpr vk::CommandBufferBeginInfo begin_info{};
    command_buffer.begin(begin_info);
    std::invoke(packaged_terrain, command_buffer);
    command_buffer.end();

    const vk::SubmitInfo submit_info{
        .commandBufferCount = 1,
        .pCommandBuffers    = &command_buffer,
    };
    vk::UniqueFence fence{ t_device->createFenceUnique({}) };

    static_cast<void>(static_cast<vk::Queue>(
                          t_device.info().get_queue(vkb::QueueType::graphics).value()
    )
                          .submit(1, &submit_info, fence.get()));

    static_cast<void>(
        t_device->waitForFences(std::array{ fence.get() }, vk::True, 100'000'000'000)
    );
    t_device->resetCommandPool(transfer_command_pool.get());

    return packaged_terrain.get_future().get();
}

[[nodiscard]]
static auto create_camera_buffer(const core::renderer::Allocator& t_allocator
) -> core::renderer::MappedBuffer
{
    constexpr vk::BufferCreateInfo buffer_create_info = {
        .size  = sizeof(core::graphics::Camera),
        .usage = vk::BufferUsageFlagBits::eUniformBuffer
    };

    return t_allocator.allocate_mapped_buffer(buffer_create_info);
}

[[nodiscard]]
static auto create_descriptor_set_layout(const vk::Device t_device
) -> vk::UniqueDescriptorSetLayout
{
    constexpr static std::array bindings{
        // Camera
        vk::DescriptorSetLayoutBinding{
                                       .binding         = 0,
                                       .descriptorType  = vk::DescriptorType::eUniformBuffer,
                                       .descriptorCount = 1,
                                       .stageFlags      = vk::ShaderStageFlagBits::eVertex
                                       | vk::ShaderStageFlagBits::eFragment },
        // Vertex buffer
        vk::DescriptorSetLayoutBinding{
                                       .binding         = 1,
                                       .descriptorType  = vk::DescriptorType::eUniformBuffer,
                                       .descriptorCount = 1,
                                       .stageFlags      = vk::ShaderStageFlagBits::eVertex
                                       | vk::ShaderStageFlagBits::eFragment },
        // Heightmap
        vk::DescriptorSetLayoutBinding{
                                       .binding         = 2,
                                       .descriptorType  = vk::DescriptorType::eSampledImage,
                                       .descriptorCount = 1,
                                       .stageFlags      = vk::ShaderStageFlagBits::eVertex
                                       | vk::ShaderStageFlagBits::eFragment },
    };

    constexpr static vk::DescriptorSetLayoutCreateInfo create_info{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data()
    };

    return t_device.createDescriptorSetLayoutUnique(create_info);
}

[[nodiscard]]
static auto create_descriptor_set(
    const vk::Device              t_device,
    const vk::DescriptorSetLayout t_descriptor_set_layout,
    const vk::DescriptorPool      t_descriptor_pool,
    const vk::Buffer              t_camera_uniform,
    const vk::Buffer              t_vertex_uniform,
    const vk::ImageView           t_heightmap_image_view,
    const vk::Sampler             t_heightmap_sampler
) -> vk::UniqueDescriptorSet
{
    const vk::DescriptorSetAllocateInfo descriptor_set_allocate_info{
        .descriptorPool     = t_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &t_descriptor_set_layout,
    };
    auto descriptor_sets{
        t_device.allocateDescriptorSetsUnique(descriptor_set_allocate_info)
    };

    const vk::DescriptorBufferInfo camera_buffer_info{
        .buffer = t_camera_uniform,
        .range  = sizeof(core::graphics::Camera),
    };
    const vk::DescriptorBufferInfo vertex_buffer_info{
        .buffer = t_vertex_uniform,
        .range  = sizeof(vk::DeviceAddress),
    };
    const vk::DescriptorImageInfo heightmap_image_info{
        .sampler     = t_heightmap_sampler,
        .imageView   = t_heightmap_image_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    std::array write_descriptor_sets{
        vk::WriteDescriptorSet{
                               .dstSet          = descriptor_sets.front().get(),
                               .dstBinding      = 0,
                               .descriptorCount = 1,
                               .descriptorType  = vk::DescriptorType::eUniformBuffer,
                               .pBufferInfo     = &camera_buffer_info,
                               },
        vk::WriteDescriptorSet{
                               .dstSet          = descriptor_sets.front().get(),
                               .dstBinding      = 1,
                               .descriptorCount = 1,
                               .descriptorType  = vk::DescriptorType::eUniformBuffer,
                               .pBufferInfo     = &vertex_buffer_info,
                               },
        vk::WriteDescriptorSet{
                               .dstSet          = descriptor_sets.front().get(),
                               .dstBinding      = 2,
                               .descriptorCount = 1,
                               .descriptorType  = vk::DescriptorType::eSampledImage,
                               .pImageInfo      = &heightmap_image_info,
                               },
    };

    t_device.updateDescriptorSets(
        static_cast<uint32_t>(write_descriptor_sets.size()),
        write_descriptor_sets.data(),
        0,
        nullptr
    );

    return std::move(descriptor_sets.front());
}

[[nodiscard]]
static auto create_pipeline_layout(
    const vk::Device                               t_device,
    const std::span<const vk::DescriptorSetLayout> t_layouts
) -> vk::UniquePipelineLayout
{
    const vk::PipelineLayoutCreateInfo pipeline_layout_create_info{
        .setLayoutCount = static_cast<uint32_t>(t_layouts.size()),
        .pSetLayouts    = t_layouts.data(),
    };

    return t_device.createPipelineLayoutUnique(pipeline_layout_create_info);
}

auto MeshRenderer::create_dependency_provider()
    -> std::shared_ptr<plugins::Renderer::DependencyProvider>
{
    return std::make_shared<DependencyProvider>();
}

auto MeshRenderer::create(Store& t_store) -> std::optional<MeshRenderer>
{
    const auto& window{ t_store.at<window::Window>() };
    auto&       device{ t_store.at<renderer::Device>() };
    auto&       allocator{ t_store.at<renderer::Allocator>() };

    auto& swapchain{ t_store.at<renderer::Swapchain>() };
    int   width{};
    int   height{};
    glfwGetFramebufferSize(window.get(), &width, &height);
    swapchain.set_framebuffer_size(vk::Extent2D{ static_cast<uint32_t>(width),
                                                 static_cast<uint32_t>(height) });
    if (!swapchain.get()) {
        return std::nullopt;
    }
    const auto& raw_swapchain{ swapchain.get().value() };

    auto render_pass{ init::create_render_pass(raw_swapchain.surface_format(), device) };
    if (!render_pass) {
        return std::nullopt;
    }

    auto depth_image{ init::create_depth_image(
        device.physical_device(), allocator, raw_swapchain.extent()
    ) };
    if (!depth_image.get()) {
        return std::nullopt;
    }

    auto depth_image_view{ init::create_depth_image_view(device, depth_image.get()) };
    if (!depth_image_view) {
        return std::nullopt;
    }

    auto framebuffers{ init::create_framebuffers(
        device.get(),
        raw_swapchain.extent(),
        raw_swapchain.image_views(),
        render_pass.get(),
        depth_image_view.get()
    ) };
    if (framebuffers.empty()) {
        return std::nullopt;
    }

    auto command_pool{ init::create_command_pool(
        device.get(), device.info().get_queue_index(vkb::QueueType::graphics).value()
    ) };
    if (!command_pool) {
        return std::nullopt;
    }

    auto command_buffers{
        init::create_command_buffers(device.get(), command_pool.get(), g_frame_count)
    };
    if (command_buffers.empty()) {
        return std::nullopt;
    }

    auto image_acquired_semaphores{ init::create_semaphores(device.get(), g_frame_count) };
    if (image_acquired_semaphores.empty()) {
        return std::nullopt;
    }

    auto render_finished_semaphores{
        init::create_semaphores(device.get(), g_frame_count)
    };
    if (render_finished_semaphores.empty()) {
        return std::nullopt;
    }

    auto in_flight_fences{ init::create_fences(device.get(), g_frame_count) };
    if (in_flight_fences.empty()) {
        return std::nullopt;
    }

    auto camera_uniform{ create_camera_buffer(allocator) };

    auto terrain{ load_terrain(device, allocator) };

    vk::UniqueDescriptorSetLayout descriptor_set_layout{
        create_descriptor_set_layout(device.get())
    };

    auto pipeline_layout{
        create_pipeline_layout(device.get(), std::array{ descriptor_set_layout.get() })
    };

    core::renderer::DescriptorPool descriptor_pool{
        core::renderer::DescriptorPool::create()
            .request_descriptor_sets(1)
            .request_descriptors(std::array{
                                            // Camera
                vk::DescriptorPoolSize{
                    .type            = vk::DescriptorType::eUniformBuffer,
                    .descriptorCount = 1,
                },   // Vertices
                vk::DescriptorPoolSize{
                    .type            = vk::DescriptorType::eUniformBuffer,
                    .descriptorCount = 1,
                },   // Heightmap
                vk::DescriptorPoolSize{
                    .type            = vk::DescriptorType::eSampledImage,
                    .descriptorCount = 2,
                }, }
            )
            .build(device.get())
    };

    vk::UniqueDescriptorSet descriptor_set{ create_descriptor_set(
        device.get(),
        descriptor_set_layout.get(),
        descriptor_pool.get(),
        camera_uniform.get(),
        terrain.vertex_uniform().get(),
        terrain.heightmap_image_view().get(),
        terrain.heightmap_sampler().get()
    ) };

    return MeshRenderer{
        .device                     = device,
        .allocator                  = allocator,
        .swapchain                  = swapchain,
        .render_pass                = std::move(render_pass),
        .depth_image                = std::move(depth_image),
        .depth_image_view           = std::move(depth_image_view),
        .framebuffers               = std::move(framebuffers),
        .command_pool               = std::move(command_pool),
        .command_buffers            = std::move(command_buffers),
        .image_acquired_semaphores  = std::move(image_acquired_semaphores),
        .render_finished_semaphores = std::move(render_finished_semaphores),
        .in_flight_fences           = std::move(in_flight_fences),
        .camera_uniform             = std::move(camera_uniform),
        .terrain                    = std::move(terrain),
        .descriptor_set_layout      = std::move(descriptor_set_layout),
        .pipeline_layout            = std::move(pipeline_layout),
        .descriptor_pool            = std::move(descriptor_pool),
        .descriptor_set             = std::move(descriptor_set),
    };
}

auto MeshRenderer::render(
    const vk::Extent2D            t_framebuffer_size,
    const core::graphics::Camera& t_camera
) -> void
{
    swapchain.get().set_framebuffer_size(t_framebuffer_size);

    while (device.get()->waitForFences(
               { in_flight_fences[frame_index].get() }, vk::True, UINT64_MAX
           )
           == vk::Result::eTimeout)
    {}

    if (auto&& [image_index, raw_swapchain]{ std::make_tuple(
            swapchain.get().acquire_next_image(
                image_acquired_semaphores[frame_index].get(), {}
            ),
            std::cref(swapchain.get())
        ) };
        image_index.has_value() && raw_swapchain.get().has_value())
    {
        device.get()->resetFences({ in_flight_fences[frame_index].get() });
        command_buffers[frame_index].reset();

        record_command_buffer(raw_swapchain.get().value(), image_index.value(), t_camera);

        std::array wait_semaphores{ image_acquired_semaphores[frame_index].get() };
        std::array<vk::PipelineStageFlags, wait_semaphores.size()> wait_stages{
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };
        std::array signal_semaphores{ render_finished_semaphores[frame_index].get() };
        const vk::SubmitInfo submit_info{
            .waitSemaphoreCount   = static_cast<uint32_t>(wait_semaphores.size()),
            .pWaitSemaphores      = wait_semaphores.data(),
            .pWaitDstStageMask    = wait_stages.data(),
            .commandBufferCount   = 1,
            .pCommandBuffers      = &command_buffers[frame_index],
            .signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size()),
            .pSignalSemaphores    = signal_semaphores.data()
        };
        vk::Queue(device.get().info().get_queue(vkb::QueueType::graphics).value())
            .submit(submit_info, in_flight_fences[frame_index].get());

        swapchain.get().present(signal_semaphores);
    }

    frame_index = (frame_index + 1) % g_frame_count;
}

auto MeshRenderer::record_command_buffer(
    const renderer::vulkan::Swapchain& t_swapchain,
    const uint32_t                     t_image_index,
    core::graphics::Camera             t_camera
) -> void
{
    const auto                           command_buffer = command_buffers[frame_index];
    constexpr vk::CommandBufferBeginInfo command_buffer_begin_info{};

    static_cast<void>(command_buffer.begin(command_buffer_begin_info));


    const std::array clear_values{
        vk::ClearValue{
                       .color = { std::array{ 0.01f, 0.01f, 0.01f, 0.01f } },
                       },
        vk::ClearValue{
                       .depthStencil = { 1.f, 0 },
                       }
    };

    const auto extent{ t_swapchain.extent() };
    command_buffer.setViewport(
        0,
        vk::Viewport{ .width    = static_cast<float>(extent.width),
                      .height   = static_cast<float>(extent.height),
                      .maxDepth = 1.f }
    );
    command_buffer.setScissor(0, vk::Rect2D{ {}, extent });

    const vk::RenderPassBeginInfo render_pass_begin_info{
        .renderPass      = render_pass.get(),
        .framebuffer     = framebuffers[t_image_index].get(),
        .renderArea      = { .extent = t_swapchain.extent() },
        .clearValueCount = static_cast<uint32_t>(clear_values.size()),
        .pClearValues    = clear_values.data()
    };
    command_buffer.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);

    t_camera.set_perspective_projection(
        50.f,
        static_cast<float>(extent.width) / static_cast<float>(extent.height),
        0.1f,
        10000.f
    );


    command_buffer.endRenderPass();
    command_buffer.end();
}
