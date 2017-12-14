#pragma once
#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <chrono>
#include <vector>
#include <opencv2/core/cvdef.h>

namespace cv {
	class Mat;
}

struct Place;

class dnnClassifier
{
private:
	using net_type = dlib::loss_multiclass_log<
		dlib::fc<10,
		dlib::relu<dlib::fc<84,
		dlib::relu<dlib::fc<120,
		dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<16, 5, 5, 1, 1,
		dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<6, 5, 5, 1, 1,
		dlib::input<dlib::matrix<unsigned char>>
		>>>>>>>>>>>>;

	net_type net;
	std::string geometryDataPath;
	std::string trainVectorPath;
	std::string svmTrainedDataPath;
	void generateTrainVector(std::string &filesListPath, float expectedResult, bool clear, std::vector<dlib::matrix<float>> &trainingImg, std::vector<float> &trainLabels);
	void extractFeatures(Place &parkingPlace, std::vector<dlib::matrix<float>> &features);
	float featureCanny(cv::Mat &place, uchar t1, uchar t2);
	void persistTrainVector(std::vector<std::vector<float>> &samples, std::vector<float> &results, bool clear);
public:
	dnnClassifier(std::string geometryDataPath, std::string trainVectorPath, std::string svmTrainedDataPath)
		: geometryDataPath(std::move(geometryDataPath)), trainVectorPath(std::move(trainVectorPath)), svmTrainedDataPath(
			std::move(svmTrainedDataPath)) {}
	void train(std::string posFilesListPath, std::string negFilesListPath);
	void recognize(std::string testFilesListPath, std::string resultsFilePath);
	void evaluate(std::string groundTruthFile, std::string resultsFilePath);
};
