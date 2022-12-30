#pragma once

#include <utility>
#include <type_traits>
#include <concepts>


template<class Product>
class BuilderBase {
    friend Product;

protected:
    constexpr [[nodiscard]] BuilderBase() noexcept = default;
    constexpr [[nodiscard]] BuilderBase(const BuilderBase&) = delete;
    constexpr [[nodiscard]] BuilderBase(BuilderBase&&) noexcept = default;

    template<typename... Args>
    constexpr explicit [[nodiscard]] BuilderBase(Args&&... args) noexcept
        : product{ std::forward<Args>(args)... } {}

    [[nodiscard]] constexpr auto draft() -> Product& {
        return product;
    }

public:
    constexpr explicit(false) [[nodiscard]] operator Product() noexcept {
        return build();
    }
    constexpr [[nodiscard]] auto build() noexcept {
        return std::move(product);
    }

private:
    Product product;
};
