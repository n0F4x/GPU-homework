#pragma once

#include <core/renderer/base/allocator/Allocator.hpp>
#include <core/renderer/base/device/Device.hpp>
#include <core/renderer/base/swapchain/Swapchain.hpp>
#include <core/renderer/memory/Image.hpp>
#include <core/renderer/scene/Scene.hpp>
#include <plugins/Renderer.hpp>

struct MeshRenderer {
    std::reference_wrapper<core::renderer::Device>    device;
    std::reference_wrapper<core::renderer::Allocator> allocator;
    std::reference_wrapper<core::renderer::Swapchain> swapchain;
    vk::UniqueRenderPass                              render_pass;
    core::renderer::Image                             depth_image;
    vk::UniqueImageView                               depth_image_view;
    std::vector<vk::UniqueFramebuffer>                framebuffers;
    vk::UniqueCommandPool                             command_pool;
    std::vector<vk::CommandBuffer>                    command_buffers;
    std::vector<vk::UniqueSemaphore>                  image_acquired_semaphores;
    std::vector<vk::UniqueSemaphore>                  render_finished_semaphores;
    std::vector<vk::UniqueFence>                      in_flight_fences;
    uint32_t                                          frame_index{};

    [[nodiscard]]
    static auto create_dependency_provider()
        -> std::shared_ptr<plugins::Renderer::DependencyProvider>;

    [[nodiscard]]
    static auto create(Store& t_store) -> std::optional<MeshRenderer>;

    auto render(vk::Extent2D t_framebuffer_size, const core::graphics::Camera& t_camera)
        -> void;

private:
    auto record_command_buffer(
        const core::renderer::vulkan::Swapchain& t_swapchain,
        uint32_t                                 t_image_index,
        core::graphics::Camera                   t_camera
    ) -> void;
};
