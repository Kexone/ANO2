#include "Loader.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cassert>
#include <iostream>
#include <sstream>

void Loader::loadParkingPlaces(std::string fileName, cv::Mat &scene, std::vector<Place> &places) {
	std::ifstream ifs;
	ifs.open(fileName);
	assert(ifs.is_open());

	while (!ifs.eof()) {
		std::string line, tmp;
		ifs >> line;
		unsigned long pos = line.find("->");

		if (pos != std::string::npos) {
			std::string el, coords = line.substr(pos + 2);
			std::istringstream is(coords);
			Place place;
			int i = 0;

			// Extract coordinates for each of 4 points
			while (getline(is, el, ';')) {
				pos = el.find(',');
				place.coords[i++] = std::stoi(el.substr(0, pos));
				place.coords[i++] = std::stoi(el.substr(pos + 1, std::string::npos));
			}

			// We should always load 8 coordinates
			assert(i == 8);

			extractFrame(scene, place);
			places.emplace_back(place);
		}
	}

	ifs.close();
}

void Loader::extractFrame(cv::Mat &scene, Place &place) {
	const cv::Size placesSize(160, 176);

	cv::Mat srcMat(4, 2, CV_32F);
	srcMat.at<float>(0, 0) = place.x1;
	srcMat.at<float>(0, 1) = place.y1;
	srcMat.at<float>(1, 0) = place.x2;
	srcMat.at<float>(1, 1) = place.y2;
	srcMat.at<float>(2, 0) = place.x3;
	srcMat.at<float>(2, 1) = place.y3;
	srcMat.at<float>(3, 0) = place.x4;
	srcMat.at<float>(3, 1) = place.y4;

	cv::Mat dstMat(4, 2, CV_32F), outMat(placesSize, CV_8UC3);
	dstMat.at<float>(0, 0) = 0;
	dstMat.at<float>(0, 1) = 0;
	dstMat.at<float>(1, 0) = outMat.cols;
	dstMat.at<float>(1, 1) = 0;
	dstMat.at<float>(2, 0) = outMat.cols;
	dstMat.at<float>(2, 1) = outMat.rows;
	dstMat.at<float>(3, 0) = 0;
	dstMat.at<float>(3, 1) = outMat.rows;

	// Checks
	assert(dstMat.type() == CV_32FC1);
	assert(srcMat.type() == CV_32FC1);
	assert(outMat.type() == CV_8UC3);

	// Extract each parking lod into rectangular matrix
	cv::Mat homography = cv::findHomography(srcMat, dstMat, 0);
	cv::warpPerspective(scene, outMat, homography, placesSize);
	place.frame = outMat;
}