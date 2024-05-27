#include "DependencyProvider.hpp"

[[nodiscard]]
auto DependencyProvider::required_instance_settings_are_available(
    const vkb::SystemInfo& t_system_info
) -> bool
{
    return t_system_info.is_extension_available(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    );
}

auto DependencyProvider::enable_instance_settings(
    const vkb::SystemInfo&,
    vkb::InstanceBuilder& t_builder
) -> void
{
    t_builder.require_api_version(1, 1);
    t_builder.enable_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
}

auto DependencyProvider::require_device_settings(
    vkb::PhysicalDeviceSelector& t_physical_device_selector
) -> void
{
    t_physical_device_selector.add_required_extension(
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
    );
    t_physical_device_selector.add_required_extension(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    t_physical_device_selector.add_required_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    t_physical_device_selector.add_required_extension_features(
        vk::PhysicalDeviceMeshShaderFeaturesEXT{
            .meshShader = vk::True,
        }
    );
}

auto DependencyProvider::enable_optional_device_settings(vkb::PhysicalDevice&) -> void {}
