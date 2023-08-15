#pragma once

#include <optional>

#include <vulkan/vulkan.hpp>

#include "engine/utility/vulkan/DebugUtilsMessenger.hpp"
#include "engine/utility/vulkan/Device.hpp"
#include "engine/utility/vulkan/Instance.hpp"

namespace engine::renderer {

class RenderDevice {
public:
    ///----------------///
    /// Static methods ///
    ///----------------///
    [[nodiscard]] static auto create(
        vulkan::Instance&&                         t_instance,
        std::optional<vulkan::DebugUtilsMessenger> t_messenger,
        vk::SurfaceKHR                             t_surface
    ) noexcept -> std::optional<RenderDevice>;

    ///------------------------------///
    ///  Constructors / Destructors  ///
    ///------------------------------///
    RenderDevice(RenderDevice&&) noexcept = default;
    ~RenderDevice() noexcept;

    ///-------------///
    ///  Operators  ///
    ///-------------///
    auto operator=(RenderDevice&&) noexcept -> RenderDevice& = default;
    [[nodiscard]] auto operator*() const noexcept -> vk::Device;
    [[nodiscard]] auto operator->() const noexcept -> const vk::Device*;

    ///-----------///
    ///  Methods  ///
    ///-----------///
    [[nodiscard]] auto physical_device() const noexcept -> vk::PhysicalDevice;
    [[nodiscard]] auto graphics_queue_family_index() const noexcept -> uint32_t;
    [[nodiscard]] auto present_queue_family_index() const noexcept -> uint32_t;

private:
    ///*************///
    ///  Variables  ///
    ///*************///
    vulkan::Instance                           m_instance;
    std::optional<vulkan::DebugUtilsMessenger> m_messenger;
    vk::PhysicalDevice                         m_physical_device;
    uint32_t                                   m_graphics_family_index;
    uint32_t                                   m_present_family_index;
    vulkan::Device                             m_device;

    ///******************************///
    ///  Constructors / Destructors  ///
    ///******************************///
    explicit RenderDevice(
        vulkan::Instance&&                         t_instance,
        std::optional<vulkan::DebugUtilsMessenger> t_messenger,
        vk::PhysicalDevice                         t_physical_device,
        uint32_t                                   t_graphics_family_index,
        uint32_t                                   t_present_family_index,
        vulkan::Device&&                           t_device
    ) noexcept;
};

}   // namespace engine::renderer
