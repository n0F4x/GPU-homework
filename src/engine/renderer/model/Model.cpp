#include "Model.hpp"

#include <glm/gtx/quaternion.hpp>

namespace engine::renderer {

auto Model::Node::local_matrix() const -> glm::mat4
{
    return glm::translate(glm::mat4(1.f), translation) * glm::toMat4(rotation)
         * glm::scale(glm::mat4(1.f), scale);
}

auto Model::Node::matrix() const -> glm::mat4
{
    glm::mat4 result{ local_matrix() };
    for (Node* p{ parent }; p != nullptr; p = p->parent) {
        result = p->local_matrix() * result;
    }
    return result;
}

}   // namespace engine::renderer