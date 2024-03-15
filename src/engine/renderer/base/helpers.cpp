#include "helpers.hpp"

#include <ranges>

#include "engine/utility/vulkan/tools.hpp"

[[nodiscard]] static auto find_graphics_queue_family(
    vk::PhysicalDevice t_physical_device,
    vk::SurfaceKHR     t_surface
) -> tl::optional<uint32_t>
{
    uint32_t index{};
    for (const auto& properties : t_physical_device.getQueueFamilyProperties()) {
        if (t_physical_device.getSurfaceSupportKHR(index, t_surface)
            && properties.queueFlags & vk::QueueFlagBits::eGraphics)
        {
            return index;
        }
        index++;
    }
    return tl::nullopt;
}

[[nodiscard]] static auto find_transfer_queue_family(
    vk::PhysicalDevice t_physical_device,
    uint32_t           t_graphics_queue_family
) -> tl::optional<uint32_t>
{
    const auto queue_families{ t_physical_device.getQueueFamilyProperties() };

    uint32_t index{};
    for (const auto& properties : queue_families) {
        if (!(properties.queueFlags & vk::QueueFlagBits::eGraphics)
            && !(properties.queueFlags & vk::QueueFlagBits::eCompute)
            && properties.queueFlags & vk::QueueFlagBits::eTransfer)
        {
            return index;
        }
        index++;
    }

    index = 0;
    for (const auto& properties : queue_families) {
        if (index == t_graphics_queue_family
            && properties.queueFlags & vk::QueueFlagBits::eTransfer)
        {
            return index;
        }
        index++;
    }

    index = 0;
    for (const auto& properties : queue_families) {
        if (properties.queueFlags & vk::QueueFlagBits::eTransfer) {
            return index;
        }
        index++;
    }

    return tl::nullopt;
}

[[nodiscard]] static auto find_compute_queue_family(
    vk::PhysicalDevice t_physical_device,
    uint32_t           t_graphics_queue_family
) noexcept -> tl::optional<uint32_t>
{
    const auto queue_families{ t_physical_device.getQueueFamilyProperties() };

    uint32_t index{};
    for (const auto& properties : queue_families) {
        if (!(properties.queueFlags & vk::QueueFlagBits::eGraphics)
            && properties.queueFlags & vk::QueueFlagBits::eCompute)
        {
            return index;
        }
        index++;
    }

    index = 0;
    for (const auto& properties : queue_families) {
        if (index != t_graphics_queue_family
            && properties.queueFlags & vk::QueueFlagBits::eCompute)
        {
            return index;
        }
        index++;
    }

    index = 0;
    for (const auto& properties : queue_families) {
        if (properties.queueFlags & vk::QueueFlagBits::eCompute) {
            return index;
        }
        index++;
    }

    return tl::nullopt;
}

namespace engine::renderer::helpers {

auto find_queue_families(vk::PhysicalDevice t_physical_device, vk::SurfaceKHR t_surface)
    -> tl::optional<QueueInfos>
{
    const auto graphics_family{ find_graphics_queue_family(t_physical_device, t_surface) };
    if (!graphics_family.has_value()) {
        return tl::nullopt;
    }

    const auto compute_family{
        find_compute_queue_family(t_physical_device, *graphics_family)
    };
    if (!compute_family.has_value()) {
        return tl::nullopt;
    }

    const auto transfer_family{
        find_transfer_queue_family(t_physical_device, *graphics_family)
    };
    if (!transfer_family.has_value()) {
        return tl::nullopt;
    }

    QueueInfos queue_infos;
    queue_infos.graphics_family = *graphics_family;
    queue_infos.compute_family  = *compute_family;
    queue_infos.transfer_family = *transfer_family;

    auto queue_priority{ 1.f };

    queue_infos.queue_create_infos.push_back(vk::DeviceQueueCreateInfo{
        .queueFamilyIndex = *graphics_family,
        .queueCount       = 1,
        .pQueuePriorities = &queue_priority });
    queue_infos.graphics_index = 0;

    if (compute_family == graphics_family) {
        if (t_physical_device.getQueueFamilyProperties()[queue_infos.graphics_family]
                .queueCount
            > queue_infos.queue_create_infos[0].queueCount)
        {
            queue_infos.queue_create_infos[0].queueCount++;
        }
        queue_infos.compute_index = queue_infos.queue_create_infos[0].queueCount - 1;
    }
    else {
        queue_infos.queue_create_infos.push_back(vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = *compute_family,
            .queueCount       = 1,
            .pQueuePriorities = &queue_priority });
        queue_infos.compute_index = 0;
    }

    if (transfer_family == graphics_family) {
        if (t_physical_device.getQueueFamilyProperties()[queue_infos.graphics_family]
                .queueCount
            > queue_infos.queue_create_infos[0].queueCount)
        {
            queue_infos.queue_create_infos[0].queueCount++;
        }
        queue_infos.transfer_index = queue_infos.queue_create_infos[0].queueCount - 1;
    }
    else if (transfer_family == compute_family) {
        if (t_physical_device.getQueueFamilyProperties()[queue_infos.compute_family]
                .queueCount
            > queue_infos.queue_create_infos[1].queueCount)
        {
            queue_infos.queue_create_infos[1].queueCount++;
        }
        queue_infos.transfer_index = queue_infos.queue_create_infos[1].queueCount - 1;
    }
    else {
        queue_infos.queue_create_infos.push_back(vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = *transfer_family,
            .queueCount       = 1,
            .pQueuePriorities = &queue_priority });
        queue_infos.transfer_index = 0;
    }

    return queue_infos;
}

}   // namespace engine::renderer::helpers