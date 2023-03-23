#include "rasterizer.h"
#include <fstream>
#include <sstream>

int main(int argc, char *argv[]) {

    uint32_t arg_index = 0;
    std::string scene_file;
    std::string render_mode;
    while ( arg_index < argc ) {
        if ( std::string(argv[arg_index]) == "--scene_file" && arg_index < argc - 1 ) {
            scene_file = argv[arg_index + 1];
            ++arg_index;
        }

        if ( std::string(argv[arg_index]) == "--render_mode" && arg_index < argc - 1 ) {
            render_mode = argv[arg_index + 1];
            ++arg_index;
        }
        ++arg_index;
    }

    auto parseVec = [](const std::string& item) {
        std::stringstream in(item);
        std::string vec_string;
        geometry::Vector3 vec;

        int i = 0;
        while ( std::getline(in, vec_string, '/') ) {
            std::stringstream ss(vec_string);
            ss >> vec[i];
            ++i;
        }

        return vec;
    };

    std::vector<geometry::Vector3> camera_trans;
    std::vector<std::string> mesh_lists;
    std::vector<std::vector<geometry::Vector3>> trans_lists;
    std::vector<std::string> texture_list;
    std::vector<float> opacity;

    std::ifstream in_stream(scene_file, std::ios::binary);
    if ( !in_stream ) throw std::runtime_error( "could not open file " + scene_file );
    std::string line;
    while( std::getline(in_stream, line) ) {
        std::stringstream ss(line);
        std::string item;

        ss >> item;
        if ( item == "Camera:" ) {
            std::getline(in_stream, line);
            std::stringstream ss_sub(line);

            ss_sub >> item;
            if ( item == "Trans:" ) {
                while ( ss_sub >> item ) {
                    camera_trans.push_back(parseVec(item));
                }
            }
        }
        if ( item == "Obj:" ) {
            for ( uint32_t i = 0; i < 4; ++i ) {
                std::getline(in_stream, line);
                std::stringstream ss_sub(line);

                ss_sub >> item;
                if ( item == "Trans:" ) {
                    std::vector<geometry::Vector3> trans;
                    while ( ss_sub >> item ) {
                        trans.push_back(parseVec(item));
                    }
                    trans_lists.push_back(trans);
                }
                else if ( item == "Mesh:" ) {
                    ss_sub >> item;
                    mesh_lists.push_back(item);
                }
                else if ( item == "Texture:" ) {
                    ss_sub >> item;
                    texture_list.push_back(item);
                }
                else if ( item == "Opacity:" ) {
                    float op;
                    ss_sub >> op;
                    opacity.push_back(op);
                }
            }
        }
    }

    renderer::Camera camera(camera_trans[0], camera_trans[1], camera_trans[2]);

    renderer::Rasterizer rasterizer(mesh_lists, trans_lists, texture_list, opacity, camera);

    if ( render_mode == "transparent" ) {
        rasterizer.setTransparent();
    }
    else if ( render_mode == "opaque" || render_mode.empty() ) {
        rasterizer.setOpaque();
    }
    else {
        std::cerr << "Unrecognized render mode. Render mode should be set either transparent or opaque." << std::endl;
        return 0;
    }

    try {
        rasterizer.run();
    }
    catch ( const std::runtime_error& err ) {
        std::cerr << err.what() << std::endl;
    }

}