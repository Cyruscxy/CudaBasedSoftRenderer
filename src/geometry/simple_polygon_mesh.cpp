#include "simple_polygon_mesh.h"
#include <iomanip>
#include <sstream>
#include <fstream>

namespace geometry {

SimplePolygonMesh::SimplePolygonMesh(const std::string& meshFilename, const std::string& type = " ") {
    readMeshFromFile(meshFilename, type);
}

SimplePolygonMesh::SimplePolygonMesh() = default;

void SimplePolygonMesh::readMeshFromFile(std::string filename, std::string type) {
    // only obj files are supported currently
    if ( type == "obj" ) {
        std::ifstream inStream(filename, std::ios::binary);
        if ( !inStream ) throw std::runtime_error( "couldn't open file " + filename );
        readMeshFromObjFile(inStream);
    }
    else {
        throw std::runtime_error("Did not recognize mesh file type " + type);
    }
}

namespace { // helper for parsing

class Index {
public:
    Index() {}
    Index(long long int v, long long int vt, long long int vn ) : position(v), uv(vt), normal(vn) {}

    bool operator<(const Index & i) const {
        if ( position < i.position ) return true;
        if ( position > i.position ) return false;
        if ( uv < i.uv ) return true;
        if ( uv > i.uv ) return false;
        if ( normal < i.normal ) return true;
        if ( normal > i.normal ) return false;
        return false;
    }

    long long int position = -1;
    long long int uv = -1;
    long long int normal = -1;
};

Index parseFaceIndex(const std::string& token) {
    std::stringstream in(token);
    std::string indexString;
    int indices[3] = {1, 1, 1};

    int i = 0;
    while ( std::getline(in, indexString, '/') ) {
        if ( indexString != "\\" ) {
            std::stringstream ss(indexString);
            ss >> indices[i];
            i++;
        }
    }

    return Index(indices[0] - 1, indices[1] - 1, indices[2] - 1);
}

}

void SimplePolygonMesh::readMeshFromObjFile(std::istream &in) {
    clear();

    std::vector<Vector2> coords;
    std::vector<std::vector<size_t>> polygonCoordInds;

    std::string line;
    while ( std::getline(in, line) ) {
        std::stringstream ss(line);
        std::string token;

        ss >> token;

        if ( token == "v" ) {
            Vector3 position;
            ss >> position;

            vertexCoordinates.push_back(position);
        }
        else if ( token == "vt" ) {
            float u, v;
            ss >> u >> v;
            vertexTexCoords.push_back(Vector2{u, v});
            coords.push_back(Vector2{u, v});
        }
        else if ( token == "vn" )  {
            // Do nothing
            Vector3 normal;
            ss >> normal;

            vertexNormals.push_back(normal);
        }
        else if ( token == "f" ) {
            std::vector<size_t> face(9);
            std::vector<size_t> faceCoordInds;

            uint32_t i = 0;
            while ( ss >> token ) {
                Index index = parseFaceIndex(token);
                if ( index.position < 0 ) {
                    getline(in, line);
                    size_t i = line.find_first_not_of("\t\n\v\f\r ");
                    index = parseFaceIndex(line.substr(i));
                }

                //face.push_back(index.position);
                face[i] = index.position;
                face[3 + i] = index.uv;
                face[6 + i] = index.normal;

                /*if ( index.uv != -1 ) {
                    //faceCoordInds.push_back(index.uv);
                }*/

                i += 1;
            }

            for ( uint32_t index = 0; index < 9; ++index ) {
                polygons.push_back(face[index]);
            }

            //polygons.push_back(face);
            if ( !faceCoordInds.empty() ) {
                polygonCoordInds.push_back(faceCoordInds);
            }
        }
    }

    // If we got uv coords, unpack them in to per-corner values
    if ( !polygonCoordInds.empty() ) {
        for ( std::vector<size_t>& faceCoordInd : polygonCoordInds ) {
            paramCoordinates.emplace_back();
            std::vector<Vector2>& faceCoord = paramCoordinates.back();
            for ( size_t i : faceCoordInd ) {
                if ( i < coords.size() ) faceCoord.push_back(coords[i]);
            }
        }
    }
}

void SimplePolygonMesh::writeMesh(std::string filename, std::string type) {
    if ( type == "obj" ) {
        std::ofstream outStream(filename);
        if ( !outStream ) throw std::runtime_error( "couldn't open output file " + filename);
        return writeMeshObj(outStream);
    }
    else {
        throw std::runtime_error("Write mesh file type " + type + " not supported");
    }
}

void SimplePolygonMesh::writeMeshObj(std::ostream &out) {
    // Make sure we write out at full precisioin
    out << std::setprecision(std::numeric_limits<double>::max_digits10);

    // Write header
    out << "# Mesh exported from geometry-central " << std::endl;
    out << "#  vertices: " << vertexCoordinates.size() << std::endl;
    out << "#     faces: " << polygons.size() << std::endl;
    out << std::endl;

    // Write vertices
    for ( Vector3 p : vertexCoordinates ) {
        out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
    }

    // Write texture coords (if present)
    for ( auto& coords : paramCoordinates ) {
        for ( Vector2 c : coords ) {
            out << "vt " << c.x << " " << c.y << std::endl;
        }
    }

    for ( auto& n: vertexCoordinates ) {
        out << "vn " << n.x << " " << n.y << " " << n.z << std::endl;
    }

    // Write faces
    size_t iC = 0;
    for ( uint32_t i = 0; i < polygons.size(); i += 9) {
        out << "f";

        for ( uint32_t j = 0; j < 3; j += 1 ) {
            out << " " << polygons[i + j] << "/" << polygons[i + j + 3] << "/" << polygons[i + j + 6] << " ";
        }
        out << std::endl;
    }
}

void SimplePolygonMesh::clear() {
    polygons.clear();
    vertexCoordinates.clear();
    paramCoordinates.clear();
}

}