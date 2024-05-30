#include "Terrain.hpp"

#include <system_error>

#include <core/asset/image/StbImage.hpp>
#include <core/renderer/base/allocator/Allocator.hpp>

[[nodiscard]]
static auto create_staging_buffer(
    const core::renderer::Allocator&          t_allocator,
    const void*                               t_data,
    const uint32_t                            t_size,
    const std::optional<const vk::DeviceSize> t_min_alignment = std::nullopt
) -> core::renderer::MappedBuffer
{
    if (t_data == nullptr || t_size == 0) {
        return core::renderer::MappedBuffer{};
    }

    const vk::BufferCreateInfo staging_buffer_create_info{
        .size  = t_size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
    };

    return t_min_alignment
        .transform([&](const vk::DeviceSize min_alignment) {
            return t_allocator.allocate_mapped_buffer_with_alignment(
                staging_buffer_create_info, min_alignment, t_data
            );
        })
        .value_or(t_allocator.allocate_mapped_buffer(staging_buffer_create_info, t_data));
}

template <typename T>
[[nodiscard]]
static auto create_staging_buffer(
    const core::renderer::Allocator&          t_allocator,
    const std::span<T>                        t_data,
    const std::optional<const vk::DeviceSize> t_min_alignment = std::nullopt
) -> core::renderer::MappedBuffer
{
    return create_staging_buffer(
        t_allocator,
        t_data.data(),
        static_cast<uint32_t>(t_data.size_bytes()),
        t_min_alignment
    );
}

[[nodiscard]]
static auto create_gpu_only_buffer(
    const core::renderer::Allocator&          t_allocator,
    const vk::BufferUsageFlags                t_usage_flags,
    const uint32_t                            t_size,
    const std::optional<const vk::DeviceSize> t_min_alignment = std::nullopt
) -> core::renderer::Buffer
{
    if (t_size == 0) {
        return core::renderer::Buffer{};
    }

    const vk::BufferCreateInfo buffer_create_info = {
        .size = t_size, .usage = t_usage_flags | vk::BufferUsageFlagBits::eTransferDst
    };

    return t_min_alignment
        .transform([&](const vk::DeviceSize min_alignment) {
            return t_allocator.allocate_buffer_with_alignment(
                buffer_create_info, min_alignment
            );
        })
        .value_or(t_allocator.allocate_buffer(buffer_create_info));
}

template <typename UniformBlock>
[[nodiscard]]
static auto create_buffer(const core::renderer::Allocator& t_allocator
) -> core::renderer::MappedBuffer
{
    constexpr vk::BufferCreateInfo buffer_create_info = {
        .size = sizeof(UniformBlock), .usage = vk::BufferUsageFlagBits::eUniformBuffer
    };

    return t_allocator.allocate_mapped_buffer(buffer_create_info);
}

[[nodiscard]]
static auto create_image(
    const core::renderer::Allocator& t_allocator,
    uint32_t                         t_width,
    uint32_t                         t_height,
    vk::Format                       t_format,
    vk::ImageTiling                  t_tiling,
    vk::ImageUsageFlags              t_usage
) -> core::renderer::Image
{
    const vk::ImageCreateInfo image_create_info{
        .imageType     = vk::ImageType::e2D,
        .format        = t_format,
        .extent        = vk::Extent3D{ .width = t_width, .height = t_height, .depth = 1 },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = vk::SampleCountFlagBits::e1,
        .tiling        = t_tiling,
        .usage         = t_usage,
        .sharingMode   = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    constexpr static VmaAllocationCreateInfo allocation_create_info = {
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    return t_allocator.allocate_image(image_create_info, allocation_create_info);
}

[[nodiscard]]
static auto create_image_view(
    const vk::Device t_device,
    const vk::Image  t_image,
    const vk::Format t_format
) -> vk::UniqueImageView
{
    const vk::ImageViewCreateInfo view_create_info{
        .image    = t_image,
        .viewType = vk::ImageViewType::e2D,
        .format   = t_format,
        .subresourceRange =
            vk::ImageSubresourceRange{ .aspectMask     = vk::ImageAspectFlagBits::eColor,
                                      .baseMipLevel   = 0,
                                      .levelCount     = 1,
                                      .baseArrayLayer = 0,
                                      .layerCount     = 1 },
    };

    return t_device.createImageViewUnique(view_create_info);
}

static void transition_image_layout(
    vk::CommandBuffer t_command_buffer,
    vk::Image         t_image,
    vk::ImageLayout   t_old_layout,
    vk::ImageLayout   t_new_layout
)
{
    vk::ImageMemoryBarrier barrier{
        .oldLayout           = t_old_layout,
        .newLayout           = t_new_layout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image               = t_image,
        .subresourceRange =
            vk::ImageSubresourceRange{ .aspectMask     = vk::ImageAspectFlagBits::eColor,
                                      .baseMipLevel   = 0,
                                      .levelCount     = 1,
                                      .baseArrayLayer = 0,
                                      .layerCount     = 1 },
    };
    vk::PipelineStageFlags source_stage;
    vk::PipelineStageFlags destination_stage;

    if (t_old_layout == vk::ImageLayout::eUndefined
        && t_new_layout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eNone;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        source_stage      = vk::PipelineStageFlagBits::eTopOfPipe;
        destination_stage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (t_old_layout == vk::ImageLayout::eTransferDstOptimal
             && t_new_layout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        source_stage      = vk::PipelineStageFlagBits::eTransfer;
        destination_stage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    t_command_buffer.pipelineBarrier(
        source_stage,
        destination_stage,
        vk::DependencyFlags{},
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier
    );
}

static void copy_buffer_to_image(
    vk::CommandBuffer t_command_buffer,
    vk::Buffer        t_buffer,
    vk::Image         t_image,
    vk::Extent3D      t_extent
)
{
    const vk::BufferImageCopy region{
        .imageSubresource =
            vk::ImageSubresourceLayers{ .aspectMask = vk::ImageAspectFlagBits::eColor,
                                       .layerCount = 1 },
        .imageExtent = t_extent,
    };

    t_command_buffer.copyBufferToImage(
        t_buffer, t_image, vk::ImageLayout::eTransferDstOptimal, 1, &region
    );
}

[[nodiscard]]
static auto create_sampler(const vk::Device t_device) -> vk::UniqueSampler
{
    const vk::SamplerCreateInfo sampler_create_info{
        .magFilter               = vk::Filter::eLinear,
        .minFilter               = vk::Filter::eLinear,
        .mipmapMode              = vk::SamplerMipmapMode::eNearest,
        .addressModeU            = vk::SamplerAddressMode::eRepeat,
        .addressModeV            = vk::SamplerAddressMode::eRepeat,
        .mipLodBias              = 0.f,
        .anisotropyEnable        = vk::False,
        .maxAnisotropy           = 1.f,
        .compareEnable           = vk::False,
        .minLod                  = 0.f,
        .maxLod                  = 1.f,
        .borderColor             = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = vk::False,
    };

    return t_device.createSamplerUnique(sampler_create_info);
}

auto Terrain::create_loader(
    vk::Device                       t_device,
    const core::renderer::Allocator& t_allocator
) -> std::packaged_task<Terrain(vk::CommandBuffer)>
{
    std::vector<Vertex> vertices;
    glm::u32vec2        quad_count{ 64, 64 };
    auto                vertex_count = static_cast<size_t>(quad_count.x)
                      * static_cast<size_t>(quad_count.y);
    vertices.reserve(vertex_count);
    for (uint32_t i{}; i < quad_count.x; i++) {
        for (uint32_t j{}; j < quad_count.y; j++) {
            auto x   = static_cast<float>(i);
            auto y   = static_cast<float>(j);
            auto max = static_cast<float>(vertex_count);

            vertices.push_back(Vertex{
                .position      = glm::vec2{       x,       y },
                .texture_coord = glm::vec2{ x / max, y / max },
            });
        }
    }

    auto vertex_buffer_size = static_cast<uint32_t>(std::span{ vertices }.size_bytes());
    core::renderer::MappedBuffer vertex_staging_buffer{
        create_staging_buffer(t_allocator, std::span{ vertices })
    };
    core::renderer::Buffer       vertex_buffer{ create_gpu_only_buffer(
        t_allocator,
        vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vertex_buffer_size
    ) };
    core::renderer::MappedBuffer vertex_uniform{
        create_buffer<vk::DeviceAddress>(t_allocator)
    };


    const auto image{ core::asset::StbImage::load_from_file(
        "res/heightmap.png", core::asset::StbImage::Channels::eGray
    ) };
    if (!image.has_value()) {
        throw std::system_error{ std::error_code{}, "Failed to load heightmap" };
    }

    core::renderer::MappedBuffer heightmap_staging_buffer{ create_staging_buffer(
        t_allocator, image->data(), static_cast<uint32_t>(image->size())
    ) };

    core::renderer::Image heightmap{ create_image(
        t_allocator,
        image->width(),
        image->height(),
        vk::Format::eR8Unorm,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
    ) };

    vk::UniqueImageView heightmap_view{
        create_image_view(t_device, heightmap.get(), vk::Format::eR8Unorm)
    };

    vk::UniqueSampler heightmap_sampler{ create_sampler(t_device) };


    return std::packaged_task<Terrain(vk::CommandBuffer)>{
        [device                   = t_device,
         vertex_buffer_size       = vertex_buffer_size,
         vertex_staging_buffer    = auto{ std::move(vertex_staging_buffer) },
         vertex_buffer            = auto{ std::move(vertex_buffer) },
         vertex_uniform           = auto{ std::move(vertex_uniform) },
         quad_count               = quad_count,
         heightmap_extent         = vk::Extent3D{ .width  = image->width(),
                                                  .height = image->height(),
                                                  .depth  = 1 },
         heightmap_staging_buffer = auto{ std::move(heightmap_staging_buffer) },
         heightmap                = auto{ std::move(heightmap) },
         heightmap_view           = auto{ std::move(heightmap_view) },
         heightmap_sampler        = auto{ std::move(heightmap_sampler
         ) }](vk::CommandBuffer t_transfer_command_buffer) mutable -> Terrain {
            t_transfer_command_buffer.copyBuffer(
                vertex_staging_buffer.get(),
                vertex_buffer.get(),
                std::array{ vk::BufferCopy{ .size = vertex_buffer_size } }
            );

            transition_image_layout(
                t_transfer_command_buffer,
                heightmap.get(),
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal
            );
            copy_buffer_to_image(
                t_transfer_command_buffer,
                heightmap_staging_buffer.get(),
                heightmap.get(),
                heightmap_extent
            );
            transition_image_layout(
                t_transfer_command_buffer,
                heightmap.get(),
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal
            );

            return Terrain{ device,
                            std::move(vertex_buffer),
                            std::move(vertex_uniform),
                            quad_count,
                            std::move(heightmap),
                            std::move(heightmap_view),
                            std::move(heightmap_sampler) };
        }
    };
}

[[nodiscard]]
auto Terrain::vertex_uniform() const noexcept -> const core::renderer::MappedBuffer&
{
    return m_vertex_uniform;
}

[[nodiscard]]
auto Terrain::heightmap_image_view() const noexcept -> const vk::UniqueImageView&
{
    return m_heightmap_view;
}

[[nodiscard]]
auto Terrain::heightmap_sampler() const noexcept -> const vk::UniqueSampler&
{
    return m_heightmap_sampler;
}

auto Terrain::draw(vk::CommandBuffer graphics_command_buffer) const -> void
{
    uint32_t group_count_x{ m_quad_count.x };
    uint32_t group_count_y{ m_quad_count.y };
    uint32_t group_count_z{ 1 };
    graphics_command_buffer.drawMeshTasksEXT(group_count_x, group_count_y, group_count_z);
}

Terrain::Terrain(
    const vk::Device               t_device,
    core::renderer::Buffer&&       t_vertex_buffer,
    core::renderer::MappedBuffer&& t_vertex_uniform,
    const glm::u32vec2             t_quad_count,
    core::renderer::Image&&        t_heightmap,
    vk::UniqueImageView&&          t_heightmap_view,
    vk::UniqueSampler&&            t_heightmap_sampler
) noexcept
    : m_vertex_buffer{ std::move(t_vertex_buffer) },
      m_vertex_uniform{ std::move(t_vertex_uniform) },
      m_quad_count{ t_quad_count },
      m_heightmap{ std::move(t_heightmap) },
      m_heightmap_view{ std::move(t_heightmap_view) },
      m_heightmap_sampler{ std::move(t_heightmap_sampler) }
{
    m_vertex_uniform.set(t_device.getBufferAddress(vk::BufferDeviceAddressInfo{
        .buffer = m_vertex_buffer.get(),
    }));
}
