#pragma once

#include <utility>
#include <type_traits>
#include <concepts>


template<class Product>
class BuilderBase {
    friend Product;

protected:
    [[nodiscard]] constexpr BuilderBase() noexcept = default;

    [[nodiscard]] constexpr explicit BuilderBase(auto&&... args) noexcept
        : product{ std::forward<decltype(args)>(args)... } {}

    [[nodiscard]] constexpr auto draft() -> Product& {
        return product;
    }

public:
    [[nodiscard]] constexpr explicit(false) operator Product() noexcept {
        return build();
    }
    [[nodiscard]] constexpr auto build() noexcept {
        return std::move(product);
    }

private:
    Product product;
};
