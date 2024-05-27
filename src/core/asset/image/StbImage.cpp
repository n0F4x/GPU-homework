#include "StbImage.hpp"

namespace core::asset {

auto StbImage::load_from_file(const std::filesystem::path& t_filepath
) -> std::optional<StbImage>
{
    int channels{};
    if (stbi_info(t_filepath.generic_string().c_str(), nullptr, nullptr, &channels) != 1)
    {
        return std::nullopt;
    }

    return load_from_file(t_filepath, static_cast<Channels>(channels));
}

auto StbImage::load_from_file(
    const std::filesystem::path& t_filepath,
    Channels                     t_desired_channels
) -> std::optional<StbImage>
{
    if (stbi_info(t_filepath.generic_string().c_str(), nullptr, nullptr, nullptr) != 1) {
        return std::nullopt;
    }

    int      width{};
    int      height{};
    stbi_uc* data{ stbi_load(
        t_filepath.generic_string().c_str(),
        &width,
        &height,
        nullptr,
        std::to_underlying(t_desired_channels)
    ) };

    if (data == nullptr) {
        SPDLOG_ERROR(stbi_failure_reason());
        return std::nullopt;
    }

    return StbImage{ data, width, height, t_desired_channels };
}

auto StbImage::load_from_memory(std::span<const std::uint8_t> t_data
) -> std::optional<StbImage>
{
    if (stbi_info_from_memory(
            t_data.data(), static_cast<int>(t_data.size()), nullptr, nullptr, nullptr
        )
        != 1)
    {
        return std::nullopt;
    }

    int      width{};
    int      height{};
    // TODO: request format
    stbi_uc* data{ stbi_load_from_memory(
        t_data.data(), static_cast<int>(t_data.size()), &width, &height, nullptr, STBI_rgb_alpha
    ) };

    if (data == nullptr) {
        SPDLOG_ERROR(stbi_failure_reason());
        return std::nullopt;
    }

    return StbImage{ data, width, height, Channels::eRGBA };
}

auto StbImage::data() const noexcept -> void*
{
    return m_data.get();
}

auto StbImage::size() const noexcept -> size_t
{
    return static_cast<size_t>(m_width) * static_cast<size_t>(m_height)
         * static_cast<size_t>(m_channels);
}

auto StbImage::width() const noexcept -> uint32_t
{
    return static_cast<uint32_t>(m_width);
}

auto StbImage::height() const noexcept -> uint32_t
{
    return static_cast<uint32_t>(m_height);
}

auto StbImage::depth() const noexcept -> uint32_t
{
    return 1u;
}

auto StbImage::mip_levels() const noexcept -> uint32_t
{
    return 1u;
}

auto StbImage::format() const noexcept -> vk::Format
{
    using enum Channels;
    switch (m_channels) {
        case eGray: return vk::Format::eR8Srgb;
        case eGrayAlpha: return vk::Format::eR8G8Srgb;
        case eRGB: return vk::Format::eR8G8B8Srgb;
        case eRGBA: return vk::Format::eR8G8B8A8Srgb;
        default: std::unreachable();
    }
}

StbImage::StbImage(
    stbi_uc*       t_data,
    const int      t_width,
    const int      t_height,
    const Channels t_channels
) noexcept
    : m_data{ t_data, stbi_image_free },
      m_width{ t_width },
      m_height{ t_height },
      m_channels{ t_channels }
{}

}   // namespace core::asset
