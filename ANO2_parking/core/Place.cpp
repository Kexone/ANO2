#include "Place.h"

std::ostream &operator<<(std::ostream &os, const Place &place) {
    os << "p1(" << place.x1 << ", " << place.y1 << "), "
       << "p3(" << place.x2 << ", " << place.y2 << "), "
       << "p3(" << place.x3 << ", " << place.y3 << "), "
       << "p4(" << place.x4 << ", " << place.y4 << "), "
       << "occupied: " << place.occupied;

    return os;
}
