#include "RenderFrame.hpp"

namespace engine::renderer {

auto RenderFrame::create(
    Device&      t_device,
    unsigned int t_thread_count,
    unsigned     t_frame_count
) noexcept -> std::optional<RenderFrame>
{
    if (t_frame_count == 0) {
        return std::nullopt;
    }

    std::vector<FrameData> frame_data;
    frame_data.reserve(t_frame_count);

    for (size_t i{}; i < t_frame_count; i++) {
        std::vector<ThreadData> thread_data;
        thread_data.reserve(t_thread_count);
        for (unsigned j{}; j < t_thread_count; j++) {
            if (auto command_pool{ vulkan::CommandPool::create(
                    *t_device,
                    vk::CommandPoolCreateFlagBits::eTransient,
                    t_device.graphics_queue_family_index()
                ) })
            {
                thread_data.emplace_back(
                    std::move(*command_pool),
                    std::vector<vk::CommandBuffer>{},
                    0
                );
            }
            else {
                return std::nullopt;
            }
        }

        auto [result, fence]{ t_device->createFence(vk::FenceCreateInfo{}) };
        if (result != vk::Result::eSuccess) {
            return std::nullopt;
        }

        frame_data.emplace_back(
            std::move(thread_data),
            vulkan::Fence{ *t_device, fence },
            std::vector<std::function<void()>>{}
        );
    }

    return RenderFrame{ std::move(frame_data) };
}

auto RenderFrame::reset(vk::Device t_device) noexcept -> vk::Result
{
    m_frame_index = (m_frame_index + 1) % m_frame_data.size();

    auto result{ t_device.waitForFences(
        *current_frame().fence, true, std::numeric_limits<uint64_t>::max()
    ) };
    if (result != vk::Result::eSuccess) {
        return result;
    }

    for (const auto& thread_data : current_frame().thread_data) {
        t_device.resetCommandPool(*thread_data.command_pool);
    }

    for (auto& update : current_frame().pre_updates) {
        update();
    }
    current_frame().pre_updates.clear();

    return vk::Result::eSuccess;
}

auto RenderFrame::request_command_buffer(
    vk::Device             t_device,
    vk::CommandBufferLevel t_level,
    unsigned               t_thread_id
) noexcept -> std::expected<vk::CommandBuffer, vk::Result>
{
    assert(t_thread_id < current_frame().thread_data.size());

    auto& current_thread{ current_frame().thread_data[t_thread_id] };
    if (current_thread.requested_command_buffers
        >= current_thread.command_buffers.size())
    {
        auto [result, command_buffer]{
            t_device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
                .commandPool        = *current_thread.command_pool,
                .level              = t_level,
                .commandBufferCount = 1 })
        };
        if (result != vk::Result::eSuccess) {
            return std::unexpected{ result };
        }
        current_thread.command_buffers.push_back(command_buffer.front());
    }

    return current_thread
        .command_buffers[current_thread.requested_command_buffers++];
}

RenderFrame::RenderFrame(std::vector<FrameData>&& t_frame_data)
    : m_frame_data{ std::move(t_frame_data) },
      m_frame_index{ m_frame_data.size() - 1 }
{}

auto RenderFrame::current_frame() noexcept -> RenderFrame::FrameData&
{
    return m_frame_data[m_frame_index];
}

}   // namespace engine::renderer