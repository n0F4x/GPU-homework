#pragma once

#include <array>
#include <expected>
#include <functional>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "engine/utility/Result.hpp"
#include "engine/utility/vulkan/CommandPool.hpp"
#include "engine/utility/vulkan/Fence.hpp"
#include "engine/utility/vulkan/Surface.hpp"
#include "engine/utility/vulkan/SwapChain.hpp"

#include "FrameData.hpp"
#include "RenderDevice.hpp"

namespace engine {

namespace renderer {

template <typename Func>
concept SurfaceCreator = std::is_nothrow_invocable_r_v<
    std::optional<vk::SurfaceKHR>,
    Func,
    vk::Instance,
    vk::Optional<const vk::AllocationCallbacks>>;

template <typename Func>
concept Recorder = std::is_nothrow_invocable_v<Func>;

struct CommandBufferAllocateInfo {
    vk::CommandBufferLevel level{};
    const void*            pNext{};
};

}   // namespace renderer

class Renderer {
public:
    ///------------------///
    ///  Nested classes  ///
    ///------------------///
    struct CreateInfo {
        std::string_view engine_name;
        uint32_t         engine_version{};
        std::string_view app_name;
        uint32_t         app_version{};
    };

    ///---------------------///
    ///  Static  Variables  ///
    ///---------------------///
    constexpr static uint32_t s_max_frames_in_flight{ 2 };

private:
    ///------------------------------///
    ///  Constructors / Destructors  ///
    ///------------------------------///
    explicit Renderer(
        renderer::RenderDevice&& t_render_device,
        vulkan::Surface&&        t_surface,
        std::array<renderer::FrameData, s_max_frames_in_flight>&& t_frame_data
    ) noexcept;

public:
    ///-----------///
    ///  Methods  ///
    ///-----------///
    auto set_framebuffer_size(vk::Extent2D t_framebuffer_size) noexcept -> void;

    auto allocate_command_buffer(
        const renderer::CommandBufferAllocateInfo& t_allocate_info,
        size_t                                     t_work_load
    ) noexcept -> std::expected<renderer::CommandHandle, vk::Result>;

    auto free_command_buffer(renderer::CommandHandle t_command) noexcept
        -> void;

    auto begin_frame() noexcept -> Result;
    auto end_frame() noexcept -> void;

    auto post_update() noexcept -> void;

    auto wait_idle() noexcept -> void;

private:
    auto recreate_swap_chain(vk::Extent2D t_framebuffer_size) noexcept -> void;

    auto get_command_buffer(renderer::CommandHandle t_command) const noexcept
        -> std::optional<renderer::CommandNodeInfo>;

public:
    ///----------------///
    /// Static methods ///
    ///----------------///
    [[nodiscard]] static auto create(
        CreateInfo                    t_context,
        renderer::SurfaceCreator auto t_create_surface,
        vk::Extent2D                  t_framebuffer_size,
        unsigned                      t_hardware_concurrency
    ) noexcept -> std::optional<Renderer>;

private:
    [[nodiscard]] static auto create(
        vulkan::Instance&& t_instance,
        vulkan::Surface&&  t_surface,
        vk::Extent2D       t_framebuffer_size,
        unsigned           t_hardware_concurrency
    ) noexcept -> std::optional<Renderer>;

    ///-------------///
    ///  Variables  ///
    ///-------------///
    bool                             m_in_frame{ false };
    renderer::RenderDevice           m_render_device;
    vulkan::Surface                  m_surface;
    std::optional<vulkan::SwapChain> m_swap_chain;

    std::array<renderer::FrameData, s_max_frames_in_flight> m_frame_data;
    uint32_t                                                m_frame_index{};

    static_assert(s_max_frames_in_flight > 1);
    std::array<std::vector<std::function<void()>>, s_max_frames_in_flight - 1>
        m_post_updates;
};

}   // namespace engine

#include "Renderer.inl"
