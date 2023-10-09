#include "Instance.hpp"

#include <format>

#include <spdlog/spdlog.h>

#include "helpers.hpp"

namespace engine::renderer {

auto Instance::create(const CreateInfo& t_create_info) noexcept
    -> std::expected<Instance, vk::Result>
{
    vk::InstanceCreateInfo create_info{
        .pNext             = t_create_info.next,
        .flags             = t_create_info.flags,
        .pApplicationInfo  = &t_create_info.application_info,
        .enabledLayerCount = static_cast<uint32_t>(t_create_info.layers.size()),
        .ppEnabledLayerNames = t_create_info.layers.data(),
        .enabledExtensionCount =
            static_cast<uint32_t>(t_create_info.extensions.size()),
        .ppEnabledExtensionNames = t_create_info.extensions.data()
    };

    auto [result, instance]{ vk::createInstance(create_info) };
    if (result != vk::Result::eSuccess) {
        SPDLOG_ERROR(
            "vk::createInstance failed with error code {}",
            static_cast<int>(result)
        );
        return std::unexpected{ result };
    }

    vk::DebugUtilsMessengerEXT debug_messenger{};
    if (t_create_info.create_debug_messenger) {
        debug_messenger = t_create_info.create_debug_messenger(instance);
    }

    return Instance{
        t_create_info.application_info,
        t_create_info.layers,
        t_create_info.extensions,
        utils::vulkan::Instance{instance, debug_messenger}
    };
}

auto Instance::create_default() noexcept -> std::expected<Instance, vk::Result>
{
    return create(CreateInfo{ .application_info = helpers::application_info(),
                              .layers           = helpers::layers(),
                              .extensions = helpers::instance_extensions(),
                              .create_debug_messenger =
                                  helpers::create_debug_messenger });
}

auto Instance::operator*() const noexcept -> vk::Instance
{
    return *m_instance;
}

auto Instance::operator->() const noexcept -> const vk::Instance*
{
    return m_instance.operator->();
}

auto Instance::application_info() const noexcept -> const vk::ApplicationInfo&
{
    return m_application_info;
}

auto Instance::enabled_layers() const noexcept -> std::span<const char* const>
{
    return m_layers;
}

auto Instance::enabled_extensions() const noexcept
    -> std::span<const char* const>
{
    return m_extensions;
}

Instance::Instance(
    const vk::ApplicationInfo&   t_application_info,
    std::span<const char* const> t_layers,
    std::span<const char* const> t_extensions,
    utils::vulkan::Instance&&    t_instance
) noexcept
    : m_application_info{ t_application_info },
      m_layers{ t_layers.begin(), t_layers.end() },
      m_extensions{ t_extensions.begin(), t_extensions.end() },
      m_instance{ std::move(t_instance) }
{
    auto [result, instance_version]{ vk::enumerateInstanceVersion() };
    if (result != vk::Result::eSuccess) {
        return;
    }

    std::string instance_info{};

    instance_info += "\nEnabled instance layers:";
    for (auto layer : m_layers) {
        instance_info += '\n';
        instance_info += '\t';
        instance_info += layer;
    }

    instance_info += "\nEnabled instance extensions:";
    for (auto extension : m_extensions) {
        instance_info += '\n';
        instance_info += '\t';
        instance_info += extension;
    }

    SPDLOG_INFO(
        "Found Vulkan Instance version: {}.{}.{}",
        VK_VERSION_MAJOR(instance_version),
        VK_VERSION_MINOR(instance_version),
        VK_VERSION_PATCH(instance_version)
    );

    SPDLOG_TRACE(instance_info);
}

}   // namespace engine::renderer