#pragma once

#include <vector>
#include <memory>
#include <string>
#include "Vector3.h"
#include "vector2.h"

namespace geometry {



class SimplePolygonMesh {
public:
    SimplePolygonMesh();
    SimplePolygonMesh(const std::string& meshFilename, const std::string& type);
    
    // == Mesh data
    std::vector<uint32_t> polygons;
    std::vector<Vector3> vertexCoordinates;
    std::vector<Vector3> vertexNormals;
    std::vector<Vector2> vertexTexCoords;
    std::vector<std::vector<Vector2>> paramCoordinates;

    // == Accessors
    [[nodiscard]]
    inline size_t nFaces() const { return polygons.size() / 9; }
    [[nodiscard]]
    inline size_t nVertices() const { return vertexCoordinates.size(); }
    [[nodiscard]]
    inline size_t nNormals() const { return vertexNormals.size(); }
    [[nodiscard]]
    inline size_t nTexCoords() const { return vertexTexCoords.size(); }
    [[nodiscard]]
    inline bool hasParameterization() const { return !paramCoordinates.empty(); }

    // == Mutator

    void clear();

    // == Input & Output
    void readMeshFromFile(std::string filename, std::string type = " ");

    void writeMesh(std::string filename, std::string type = "");

private:
    void readMeshFromObjFile(std::istream& in);
    void writeMeshObj(std::ostream &output);

};

}