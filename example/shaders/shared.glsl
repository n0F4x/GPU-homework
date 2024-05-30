const uint invocationCountX = 2;
const uint invocationCountY = 2;

struct SharedData {
    uint vertexIndices[4];
    uint tesselationLevel;
    vec3 offsetX;
    vec3 offsetY;
};
