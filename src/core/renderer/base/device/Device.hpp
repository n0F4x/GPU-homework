#pragma once

#include <vulkan/vulkan.hpp>

#include <VkBootstrap.h>

namespace core::renderer {

class Device {
public:
    ///------------------------------///
    ///  Constructors / Destructors  ///
    ///------------------------------///
    explicit Device(const vkb::Device& t_device);
    explicit Device(vkb::Device&& t_device) noexcept;
    Device(const Device&)     = delete;
    Device(Device&&) noexcept = default;
    ~Device();

    ///-------------///
    ///  Operators  ///
    ///-------------///
    auto operator=(const Device&)                = delete;
    auto operator=(Device&&) noexcept -> Device& = default;
    [[nodiscard]]
    auto operator*() const noexcept -> vk::Device;
    [[nodiscard]]
    auto operator->() const noexcept -> const vk::Device*;
    [[nodiscard]]
    explicit operator vkb::Device() const;

    ///-----------///
    ///  Methods  ///
    ///-----------///
    [[nodiscard]]
    auto get() const noexcept -> vk::Device;
    [[nodiscard]]
    auto physical_device() const noexcept -> vk::PhysicalDevice;
    [[nodiscard]]
    auto info() const -> vkb::Device;

private:
    ///*************///
    ///  Variables  ///
    ///*************///
    vkb::Device m_device;
};

}   // namespace core::renderer
