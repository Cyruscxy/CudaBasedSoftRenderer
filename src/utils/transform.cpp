#include "transform.h"

namespace geometry {

Mat4 Transform::local_to_world() {
    if ( std::shared_ptr<Transform> parent_ = this->parent.lock() ) {
        return parent_->local_to_world() * local_to_parent();
    }
    else {
        return local_to_parent();
    }
}

Mat4 Transform::world_to_local() {
    if ( std::shared_ptr<Transform> parent_ = this->parent.lock() ) {
        return parent_to_local() * parent_->world_to_local();
    }
    else {
        return parent_to_local();
    }
}

Mat4 Transform::local_to_parent() {
    return Mat4::translate(location) * rotation.to_mat() * Mat4::scale(scale);
}

Mat4 Transform::parent_to_local() {
    return Mat4::scale(Vector3{1.f / scale.x, 1.f / scale.y, 1.f / scale.z}) * rotation.inverse().to_mat() * Mat4::translate(-location);
}

}

