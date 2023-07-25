#include "SwapChain.hpp"

#include <algorithm>   // std::clamp
#include <limits>      // std::numeric_limits
#include <set>

namespace {

auto choose_swap_chain_surface_format(
    vk::SurfaceKHR     t_surface,
    vk::PhysicalDevice t_physical_device
) -> std::optional<vk::SurfaceFormatKHR>
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
) -> std::optional<vk::PresentModeKHR>
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
) -> vk::Extent2D
{
    if (t_surface_capabilities.currentExtent.width
        == std::numeric_limits<uint32_t>::max())
    {
        auto actual_extent = t_frame_buffer_size;

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

        return { actual_extent.width, actual_extent.height };
    }
    else {
        return t_surface_capabilities.currentExtent;
    }
}

auto create_swap_chain(
    vk::SurfaceKHR       t_surface,
    vk::PhysicalDevice   t_physical_device,
    uint32_t             t_graphics_queue_family,
    uint32_t             t_present_queue_family,
    vk::Device           t_device,
    vk::Extent2D         t_extent,
    vk::SurfaceFormatKHR t_surfaceFormat
) -> std::optional<vk::SwapchainKHR>
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
    };

    auto swap_chain{ t_device.createSwapchainKHR(create_info) };
    if (swap_chain.result != vk::Result::eSuccess) {
        return std::nullopt;
    }

    return swap_chain.value;
}

auto create_image_views(
    vk::Device           t_device,
    vk::SwapchainKHR     t_swap_chain,
    vk::SurfaceFormatKHR t_surface_format
) -> std::optional<std::vector<vk::ImageView>>
{
    auto images = t_device.getSwapchainImagesKHR(t_swap_chain);
    if (images.result != vk::Result::eSuccess) {
        return std::nullopt;
    }
    std::vector<vk::ImageView> image_views;
    image_views.reserve(images.value.size());

    vk::ImageViewCreateInfo image_view_create_info{
        .viewType         = vk::ImageViewType::e2D,
        .format           = t_surface_format.format,
        .subresourceRange = {.aspectMask     = vk::ImageAspectFlagBits::eColor,
                             .baseMipLevel   = 0,
                             .levelCount     = 1,
                             .baseArrayLayer = 0,
                             .layerCount     = 1}
    };
    for (auto image : images.value) {
        image_view_create_info.image = image;
        auto image_view{ t_device.createImageView(image_view_create_info) };
        if (image_view.result != vk::Result::eSuccess) {
            return std::nullopt;
        }
        image_views.push_back(image_view.value);
    }

    return image_views;
}

}   // namespace

namespace engine::vulkan {

////////////////////////////////////
///------------------------------///
///  SwapChain   IMPLEMENTATION  ///
///------------------------------///
////////////////////////////////////

SwapChain::SwapChain(
    Surface&&                  t_surface,
    vk::Device                 t_device,
    vk::Extent2D               t_extent,
    vk::SurfaceFormatKHR       t_surface_format,
    vk::SwapchainKHR           t_swap_chain,
    std::vector<vk::ImageView> t_image_views
) noexcept
    : m_surface{ std::move(t_surface) },
      m_device{ t_device },
      m_extent{ t_extent },
      m_surface_format{ t_surface_format },
      m_swap_chain{ t_swap_chain },
      m_imageViews{ std::move(t_image_views) }
{}

SwapChain::SwapChain(SwapChain&& t_other) noexcept
    : m_surface{ std::move(t_other.m_surface) },
      m_device{ t_other.m_device },
      m_extent{ t_other.m_extent },
      m_surface_format{ t_other.m_surface_format },
      m_swap_chain{ t_other.m_swap_chain },
      m_imageViews{ std::move(t_other.m_imageViews) }
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

auto SwapChain::image_views() const noexcept
    -> const std::vector<vk::ImageView>&
{
    return m_imageViews;
}

auto engine::vulkan::SwapChain::create(
    Surface&&          t_surface,
    vk::PhysicalDevice t_physical_device,
    uint32_t           t_graphics_queue_family,
    uint32_t           t_present_queue_family,
    vk::Device         t_device,
    vk::Extent2D       t_frame_buffer_size
) noexcept -> std::optional<engine::vulkan::SwapChain>
{
    auto surface_capabilities{
        t_physical_device.getSurfaceCapabilitiesKHR(*t_surface)
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
        choose_swap_chain_surface_format(*t_surface, t_physical_device)
    };
    if (!surface_format.has_value()) {
        return std::nullopt;
    }

    auto swap_chain{ create_swap_chain(
        *t_surface,
        t_physical_device,
        t_graphics_queue_family,
        t_present_queue_family,
        t_device,
        extent,
        *surface_format
    ) };
    if (!swap_chain.has_value()) {
        return std::nullopt;
    }

    auto image_views{
        create_image_views(t_device, *swap_chain, *surface_format)
    };
    if (!image_views.has_value()) {
        return std::nullopt;
    }

    return {
        SwapChain{std::move(t_surface),
                  t_device, extent,
                  *surface_format,
                  *swap_chain,
                  std::move(*image_views)}
    };
}

}   // namespace engine::vulkan
