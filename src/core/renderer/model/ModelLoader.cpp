#include "ModelLoader.hpp"

#include <ranges>

#include <spdlog/spdlog.h>

#include <entt/core/hashed_string.hpp>

#include <fastgltf/core.hpp>

#include "GltfLoader.hpp"

[[nodiscard]] static auto load_asset(const std::filesystem::path& t_filepath) noexcept
    -> fastgltf::Expected<fastgltf::Asset>
{
    fastgltf::Parser parser;

    constexpr auto gltf_options{ fastgltf::Options::LoadGLBBuffers
                                 | fastgltf::Options::LoadExternalBuffers
                                 | fastgltf::Options::GenerateMeshIndices
                                 | fastgltf::Options::DecomposeNodeMatrices };

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(t_filepath);

    return parser.loadGltf(&data, t_filepath.parent_path(), gltf_options);
}

namespace core::renderer {

ModelLoader::ModelLoader(Cache& t_cache) noexcept : m_cache{ t_cache } {}

auto ModelLoader::load_from_file(
    const std::filesystem::path& t_filepath,
    const renderer::Allocator&   t_allocator
) noexcept -> tl::optional<StagingModel>
{
    auto asset{ load_asset(t_filepath) };
    if (asset.error() != fastgltf::Error::None) {
        SPDLOG_ERROR("Failed to load glTF: {}", fastgltf::to_underlying(asset.error()));
        return tl::nullopt;
    }

    GltfLoader model;
    model.load(asset.get());

    return StagingMeshBuffer::create<Vertex>(t_allocator, model.vertices, model.indices)
        .transform([&](StagingMeshBuffer&& t_staging_mesh_buffer) {
            return StagingModel{ std::move(t_staging_mesh_buffer),
                                 std::move(model.nodes) };
        });
}

//auto ModelLoader::load_from_file(
//    const std::filesystem::path&   t_filepath,
//    const VertexInputStateBuilder& t_vertex_input_state
//) noexcept -> tl::optional<Handle<Model>>
//{}

}   // namespace core::renderer
