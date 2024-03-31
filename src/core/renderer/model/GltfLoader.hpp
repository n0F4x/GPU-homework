#pragma once

#include <vector>

#include <fastgltf/types.hpp>

#include "Model.hpp"
#include "StagingModel.hpp"

namespace core::renderer {

struct GltfLoader {
    std::vector<Model::Vertex>      vertices;
    std::vector<uint32_t>           indices;
    std::vector<StagingModel::Node> nodes;

    auto load(const fastgltf::Asset& t_asset) -> void;

private:
    using Primitive = StagingModel::Primitive;

    auto load_image(const fastgltf::Image&) -> void;

    auto load_node(
        StagingModel::Node*    t_parent,
        const fastgltf::Asset& t_asset,
        const fastgltf::Node&  t_node
    ) noexcept -> void;

    [[nodiscard]] auto
        load_mesh(const fastgltf::Asset& t_asset, const fastgltf::Mesh& t_mesh) noexcept
        -> StagingModel::Mesh;

    auto load_primitive(
        StagingModel::Mesh&        t_mesh,
        const fastgltf::Asset&     t_asset,
        const fastgltf::Primitive& t_primitive
    ) noexcept -> void;

    [[nodiscard]] auto load_vertices(
        Primitive&             t_primitive,
        const fastgltf::Asset& t_asset,
        const fastgltf::pmr::SmallVector<fastgltf::Primitive::attribute_type, 4>& t_attributes
    ) noexcept -> bool;

    auto load_indices(
        Primitive&                t_primitive,
        uint32_t                  t_first_vertex_index,
        const fastgltf::Asset&    t_asset,
        const fastgltf::Accessor& t_accessor
    ) noexcept -> void;
};

}   // namespace core::renderer