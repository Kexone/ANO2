#ifndef VSB_ANO2_PARKING_PLACE_H
#define VSB_ANO2_PARKING_PLACE_H

#include <ostream>
#include <opencv2/core/mat.hpp>

struct Place {
public:
    union {
        struct {
            int x1, y1, x2, y2, x3, y3, x4, y4;
        };

        int coords[8];
    };

    cv::Mat frame;
    bool occupied;

    Place() = default;
    Place(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, bool occupied = false) :
            x1(x1), y1(y1), x2(x2), y2(y2), x3(x3), y3(y3), x4(x4), y4(y4), occupied(occupied) {}

    friend std::ostream &operator<<(std::ostream &os, const Place &place);
};

#endif //VSB_ANO2_PARKING_PLACE_H
