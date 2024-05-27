#include <plugins/Renderer.hpp>

class DependencyProvider final : public plugins::Renderer::DependencyProvider {
public:
    [[nodiscard]]
    auto required_instance_settings_are_available(const vkb::SystemInfo& t_system_info
    ) -> bool override;

    auto enable_instance_settings(const vkb::SystemInfo&, vkb::InstanceBuilder& t_builder)
        -> void override;

    auto require_device_settings(vkb::PhysicalDeviceSelector& t_physical_device_selector
    ) -> void override;

    auto enable_optional_device_settings(vkb::PhysicalDevice&) -> void override;
};
