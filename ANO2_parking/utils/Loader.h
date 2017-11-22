#ifndef VSB_ANO2_PARKING_LOADER_H
#define VSB_ANO2_PARKING_LOADER_H

#include <string>
#include <vector>
#include "../core/Place.h"

class Loader {
private:
	static void extractFrame(cv::Mat &scene, Place &place);
public:
	static void loadParkingPlaces(std::string fileName, cv::Mat &scene, std::vector<Place> &places);
};

#endif //VSB_ANO2_PARKING_LOADER_H