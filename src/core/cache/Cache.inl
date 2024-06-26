namespace core::cache {

template <typename IdType, template <typename...> typename ContainerTemplate>
template <typename Resource>
auto BasicCache<IdType, ContainerTemplate>::insert(
    ID                      t_id,
    const Handle<Resource>& t_handle
) -> Handle<Resource>
{
    m_store.emplace<ContainerType<Resource>>().try_emplace(
        t_id, static_cast<std::shared_ptr<Resource>>(t_handle)
    );
    return t_handle;
}

template <typename IdType, template <typename...> typename ContainerTemplate>
template <typename Resource>
auto BasicCache<IdType, ContainerTemplate>::insert(ID t_id, Handle<Resource>&& t_handle)
    -> Handle<Resource>
{
    m_store.emplace<ContainerType<Resource>>().try_emplace(
        t_id, static_cast<std::shared_ptr<Resource>>(t_handle)
    );
    return t_handle;
}

template <typename IdType, template <typename...> typename ContainerTemplate>
template <typename Resource>
auto BasicCache<IdType, ContainerTemplate>::emplace(ID t_id, auto&&... t_args)
    -> Handle<Resource>
{
    auto result{ make_handle<Resource>(std::forward<decltype(t_args)>(t_args)...) };
    m_store.emplace<ContainerType<Resource>>().try_emplace(
        t_id, static_cast<std::shared_ptr<Resource>>(result)
    );
    return result;
}

template <typename IdType, template <typename...> typename ContainerTemplate>
template <typename Resource>
auto BasicCache<IdType, ContainerTemplate>::find(ID t_id
) const noexcept -> std::optional<Handle<Resource>>
{
    return m_store.find<ContainerType<Resource>>().and_then(
        [t_id](const ContainerType<Resource>& t_container
        ) -> std::optional<Handle<Resource>> {
            const auto iter{ t_container.find(t_id) };
            if (iter == t_container.cend()) {
                return std::nullopt;
            }
            auto result{ iter->second.lock() };
            return result != nullptr ? std::optional{ std::move(result) } : std::nullopt;
        }
    );
}

template <typename IdType, template <typename...> typename ContainerTemplate>
template <typename Resource>
auto BasicCache<IdType, ContainerTemplate>::at(ID t_id) const -> Handle<Resource>
{
    return m_store.at<ContainerType<Resource>>().at(t_id).lock();
}

template <typename IdType, template <typename...> typename ContainerTemplate>
template <typename Resource>
auto BasicCache<IdType, ContainerTemplate>::remove(ID t_id
) noexcept -> std::optional<Handle<Resource>>
{
    return m_store.find<ContainerType<Resource>>().and_then(
        [t_id](ContainerType<Resource>& t_container) -> std::optional<Handle<Resource>> {
            const auto iter{ t_container.find(t_id) };
            if (iter == t_container.cend()) {
                return std::nullopt;
            }

            t_container.erase(iter);

            return iter->second;
        }
    );
}

}   // namespace core::cache
