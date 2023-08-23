#include "SwapChain.hpp"

#include <algorithm>
#include <limits>
#include <set>

namespace {

auto choose_swap_chain_surface_format(
    vk::SurfaceKHR     t_surface,
    vk::PhysicalDevice t_physical_device
) noexcept -> std::optional<vk::SurfaceFormatKHR>
{
    auto [result, t_available_surface_formats]{
        t_physical_device.getSurfaceFormatsKHR(t_surface)
    };
    if (result != vk::Result::eSuccess) {
        return std::nullopt;
    }

    for (const auto& surface_format : t_available_surface_formats) {
        if (surface_format.format == vk::Format::eB8G8R8A8Srgb
            && surface_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return surface_format;
        }
    }
    return t_available_surface_formats.front();
}

auto choose_swap_chain_present_mode(
    vk::SurfaceKHR     t_surface,
    vk::PhysicalDevice t_physical_device
) noexcept -> std::optional<vk::PresentModeKHR>
{
    auto [result, present_modes]{
        t_physical_device.getSurfacePresentModesKHR(t_surface)
    };
    if (result != vk::Result::eSuccess) {
        return std::nullopt;
    }
    for (auto available_present_mode : present_modes) {
        if (available_present_mode == vk::PresentModeKHR::eMailbox) {
            return available_present_mode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

auto choose_swap_chain_extent(
    const vk::Extent2D&               t_frame_buffer_size,
    const vk::SurfaceCapabilitiesKHR& t_surface_capabilities
) noexcept -> vk::Extent2D
{
    if (t_surface_capabilities.currentExtent.width
        != std::numeric_limits<uint32_t>::max())
    {
        return t_surface_capabilities.currentExtent;
    }
    else {
        vk::Extent2D actual_extent{
            .width  = static_cast<uint32_t>(t_frame_buffer_size.width),
            .height = static_cast<uint32_t>(t_frame_buffer_size.height)
        };

        actual_extent.width = std::clamp(
            actual_extent.width,
            t_surface_capabilities.minImageExtent.width,
            t_surface_capabilities.maxImageExtent.width
        );
        actual_extent.height = std::clamp(
            actual_extent.height,
            t_surface_capabilities.minImageExtent.height,
            t_surface_capabilities.maxImageExtent.height
        );

        return actual_extent;
    }
}

auto create_swap_chain(
    vk::SurfaceKHR                  t_surface,
    vk::PhysicalDevice              t_physical_device,
    uint32_t                        t_graphics_queue_family,
    uint32_t                        t_present_queue_family,
    vk::Device                      t_device,
    vk::Extent2D                    t_extent,
    vk::SurfaceFormatKHR            t_surfaceFormat,
    std::optional<vk::SwapchainKHR> t_old_swap_chain
) noexcept -> std::optional<vk::SwapchainKHR>
{
    auto [result, surface_capabilities]{
        t_physical_device.getSurfaceCapabilitiesKHR(t_surface)
    };
    if (result != vk::Result::eSuccess) {
        return std::nullopt;
    }
    auto present_mode{
        choose_swap_chain_present_mode(t_surface, t_physical_device)
    };
    if (!present_mode.has_value()) {
        return std::nullopt;
    }

    uint32_t image_count = surface_capabilities.minImageCount + 1;
    if (surface_capabilities.maxImageCount > 0
        && image_count > surface_capabilities.maxImageCount)
    {
        image_count = surface_capabilities.maxImageCount;
    }

    std::set buffer{ t_graphics_queue_family, t_present_queue_family };
    std::vector<uint32_t> queueFamilyIndices = { buffer.begin(), buffer.end() };
    vk::SharingMode       sharingMode        = queueFamilyIndices.size() > 1
                                                 ? vk::SharingMode::eConcurrent
                                                 : vk::SharingMode::eExclusive;

    vk::SwapchainCreateInfoKHR create_info{
        .surface          = t_surface,
        .minImageCount    = image_count,
        .imageFormat      = t_surfaceFormat.format,
        .imageColorSpace  = t_surfaceFormat.colorSpace,
        .imageExtent      = t_extent,
        .imageArrayLayers = 1,
        .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = sharingMode,
        .queueFamilyIndexCount =
            static_cast<uint32_t>(queueFamilyIndices.size()),
        .pQueueFamilyIndices = queueFamilyIndices.data(),
        .preTransform        = surface_capabilities.currentTransform,
        .compositeAlpha      = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode         = *present_mode,
        .clipped             = true,
        .oldSwapchain        = t_old_swap_chain.value_or(nullptr)
    };

    auto swap_chain{ t_device.createSwapchainKHR(create_info) };
    if (swap_chain.result != vk::Result::eSuccess) {
        return std::nullopt;
    }

    return swap_chain.value;
}

}   // namespace

namespace engine::vulkan {

////////////////////////////////////
///------------------------------///
///  SwapChain   IMPLEMENTATION  ///
///------------------------------///
////////////////////////////////////

SwapChain::SwapChain(
    vk::Device           t_device,
    vk::Extent2D         t_extent,
    vk::SurfaceFormatKHR t_surface_format,
    vk::SwapchainKHR     t_swap_chain
) noexcept
    : m_device{ t_device },
      m_extent{ t_extent },
      m_surface_format{ t_surface_format },
      m_swap_chain{ t_swap_chain }
{}

SwapChain::SwapChain(SwapChain&& t_other) noexcept
    : m_device{ t_other.m_device },
      m_extent{ t_other.m_extent },
      m_surface_format{ t_other.m_surface_format },
      m_swap_chain{ t_other.m_swap_chain }
{
    t_other.m_swap_chain = nullptr;
}

SwapChain::~SwapChain() noexcept
{
    if (m_swap_chain) {
        m_device.destroy(m_swap_chain);
    }
}

auto SwapChain::operator*() const noexcept -> vk::SwapchainKHR
{
    return m_swap_chain;
}

auto SwapChain::extent() const noexcept -> vk::Extent2D
{
    return m_extent;
}

auto engine::vulkan::SwapChain::create(
    vk::SurfaceKHR                  t_surface,
    vk::PhysicalDevice              t_physical_device,
    uint32_t                        t_graphics_queue_family,
    uint32_t                        t_present_queue_family,
    vk::Device                      t_device,
    vk::Extent2D                    t_frame_buffer_size,
    std::optional<vk::SwapchainKHR> t_old_swap_chain
) noexcept -> std::optional<engine::vulkan::SwapChain>
{
    auto surface_capabilities{
        t_physical_device.getSurfaceCapabilitiesKHR(t_surface)
    };
    if (surface_capabilities.result != vk::Result::eSuccess) {
        return std::nullopt;
    }

    auto extent = choose_swap_chain_extent(
        t_frame_buffer_size, surface_capabilities.value
    );
    if (extent.width == 0 || extent.height == 0) {
        return std::nullopt;
    }

    auto surface_format{
        choose_swap_chain_surface_format(t_surface, t_physical_device)
    };
    if (!surface_format.has_value()) {
        return std::nullopt;
    }

    auto swap_chain{ create_swap_chain(
        t_surface,
        t_physical_device,
        t_graphics_queue_family,
        t_present_queue_family,
        t_device,
        extent,
        *surface_format,
        t_old_swap_chain
    ) };
    if (!swap_chain.has_value()) {
        return std::nullopt;
    }

    return SwapChain{ t_device, extent, *surface_format, *swap_chain };
}

}   // namespace engine::vulkan
