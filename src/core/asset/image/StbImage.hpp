#pragma once

#include <filesystem>
#include <optional>
#include <span>

#include <stb_image.h>

#include "Image.hpp"

namespace core::asset {

class StbImage final : public Image {
public:
    enum class Channels {
        eGray      = 1,
        eGrayAlpha = 2,
        eRGB       = 3,
        eRGBA      = 4
    };

    [[nodiscard]]
    static auto load_from_file(const std::filesystem::path& t_filepath
    ) -> std::optional<StbImage>;

    [[nodiscard]]
    static auto
        load_from_file(const std::filesystem::path& t_filepath, Channels desired_channels)
            -> std::optional<StbImage>;

    [[nodiscard]]
    static auto load_from_memory(std::span<const std::uint8_t> t_data
    ) -> std::optional<StbImage>;

    [[nodiscard]]
    auto data() const noexcept -> void* override;
    [[nodiscard]]
    auto size() const noexcept -> size_t override;

    [[nodiscard]]
    auto width() const noexcept -> uint32_t override;
    [[nodiscard]]
    auto height() const noexcept -> uint32_t override;
    [[nodiscard]]
    auto depth() const noexcept -> uint32_t override;

    [[nodiscard]]
    auto mip_levels() const noexcept -> uint32_t override;

    [[nodiscard]]
    auto format() const noexcept -> vk::Format override;

private:
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> m_data;
    int                                                  m_width;
    int                                                  m_height;
    Channels                                             m_channels;

    explicit StbImage(stbi_uc* data, int width, int height, Channels channels) noexcept;
};

}   // namespace core::asset
