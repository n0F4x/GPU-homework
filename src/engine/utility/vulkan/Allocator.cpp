#define VMA_IMPLEMENTATION
#include "Allocator.hpp"

namespace engine::utils::vulkan {

Allocator::Allocator(VmaAllocator t_allocator) noexcept
    : m_allocator{ t_allocator }
{}

Allocator::Allocator(Allocator&& t_other) noexcept
    : Allocator{ t_other.m_allocator }
{
    t_other.m_allocator = nullptr;
}

Allocator::~Allocator() noexcept
{
    if (m_allocator) {
        vmaDestroyAllocator(m_allocator);
    }
}

auto Allocator::operator*() const noexcept -> VmaAllocator
{
    return m_allocator;
}

auto Allocator::operator->() const noexcept -> const VmaAllocator*
{
    return &m_allocator;
}

}   // namespace engine::utils::vulkan
