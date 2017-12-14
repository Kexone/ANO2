#include "dnnClassifier.h"
#include "../utils/Loader.h"
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/shape/hist_cost.hpp>

void dnnClassifier::train(std::string posFilesListPath, std::string negFilesListPath)
{
	std::vector<dlib::matrix<float>> training_images;
	std::vector<float>         training_labels;
	
	generateTrainVector(posFilesListPath, 1.0f, true, training_images, training_labels);
	generateTrainVector(negFilesListPath, -1.0f, false ,training_images, training_labels);

	dlib::dnn_trainer<net_type> trainer(net);
	trainer.set_learning_rate(0.01);
	trainer.set_min_learning_rate(0.00001);
	trainer.set_mini_batch_size(128);
	trainer.be_verbose();

	trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));
	std::cout << "I "<< training_images.size() << std::endl;
	std::cout << "L " <<  training_labels.size() << std::endl;
 	//trainer.train(training_images, training_labels);
	dlib::serialize("trained_network.dat") << net;

}

void dnnClassifier::recognize(std::string testFilesListPath, std::string resultsFilePath)
{
	std::vector<unsigned long> predicted_labels;
	int num_right, num_wrong;
	std::vector<dlib::matrix<unsigned char>> testing_images;
	std::vector<unsigned long>         testing_labels;
	net.clean();
	dlib::deserialize(svmTrainedDataPath) >> net;

	//predicted_labels = net(training_images);
	//num_right = 0;
	//num_wrong = 0;
	//// And then let's see if it classified them correctly.
	//for (size_t i = 0; i < training_images.size(); ++i)
	//{
	//	if (predicted_labels[i] == training_labels[i])
	//		++num_right;
	//	else
	//		++num_wrong;

	//}
	//std::cout << "training num_right: " << num_right << std::endl;
	//std::cout << "training num_wrong: " << num_wrong << std::endl;
	//std::cout << "training accuracy:  " << num_right / (double)(num_right + num_wrong) << std::endl;

	predicted_labels = net(testing_images);
	num_right = 0;
	num_wrong = 0;
	for (size_t i = 0; i < testing_images.size(); ++i)
	{
		if (predicted_labels[i] == testing_labels[i])
			++num_right;
		else
			++num_wrong;

	}
	std::cout << "testing num_right: " << num_right << std::endl;
	std::cout << "testing num_wrong: " << num_wrong << std::endl;
	std::cout << "testing accuracy:  " << num_right / (double)(num_right + num_wrong) << std::endl;


	// Finally, you can also save network parameters to XML files if you want to do
	// something with the network in another tool.  For example, you could use dlib's
	// tools/convert_dlib_nets_to_caffe to convert the network to a caffe model.
	//dlib::net_to_xml(net, "lenet.xml");
}

void dnnClassifier::evaluate(std::string groundTruthFile, std::string resultsFilePath)
{
	// Load files
	std::ifstream groundTruthStream, resultsStream;
	std::ofstream file;
	groundTruthStream.open(groundTruthFile);
	resultsStream.open(resultsFilePath);
	assert(groundTruthStream.is_open());
	assert(resultsStream.is_open());

	int detectorLine, groundTruthLine;
	int falsePositives = 0;
	int falseNegatives = 0;
	int truePositives = 0;
	int trueNegatives = 0;

	while (true) {
		if (!(resultsStream >> detectorLine)) break;
		groundTruthStream >> groundTruthLine;

		int detect = detectorLine;
		int ground = groundTruthLine;

		//false positives
		if ((detect == 1) && (ground == 0)) {
			falsePositives++;
		}

		//false negatives
		if ((detect == 0) && (ground == 1)) {
			falseNegatives++;
		}

		//true positives
		if ((detect == 1) && (ground == 1)) {
			truePositives++;
		}

		//true negatives
		if ((detect == 0) && (ground == 0)) {
			trueNegatives++;
		}
	}

	groundTruthStream.close();
	resultsStream.close();

	file.open("resultTrain.txt", std::ios::app);

	std::cout << "falsePositives " << falsePositives << std::endl;
	std::cout << "falseNegatives " << falseNegatives << std::endl;
	std::cout << "truePositives " << truePositives << std::endl;
	std::cout << "trueNegatives " << trueNegatives << std::endl;
	float acc = (float)(truePositives + trueNegatives) /
		(float)(truePositives + trueNegatives + falsePositives + falseNegatives);
	std::cout << "Accuracy " << acc << std::endl;

	file << "falsePositives " << falsePositives << std::endl;
	file << "falseNegatives " << falseNegatives << std::endl;
	file << "truePositives " << truePositives << std::endl;
	file << "trueNegatives " << trueNegatives << std::endl;
	file << "Accuracy " << acc << std::endl;

	file.close();
	//return acc;
}


void dnnClassifier::generateTrainVector(std::string &filesListPath, float expectedResult, bool clear, std::vector<dlib::matrix<float>> &trainingImg, std::vector<float> &trainLabels) {
	// Open file list
	std::ifstream ofs;
	ofs.open(filesListPath);
	assert(ofs.is_open());

	bool first = clear; // Clear file on first pass
	std::vector<std::vector<float>> samples;
	std::vector<float> results, features;


	// Extract features for each file
	while (!ofs.eof()) {
		std::string line;
		ofs >> line;

		std::cout << "Loading: " << line << std::endl;
		cv::Mat scene = cv::imread(line, CV_LOAD_IMAGE_COLOR);

		// Load places geometry
		std::vector<Place> places;
		Loader::loadParkingPlaces(geometryDataPath, scene, places);

		for (auto &parkingPlace : places) {
			extractFeatures(parkingPlace, trainingImg);
			//trainingImg.push_back(features);
			results.push_back(expectedResult);
			features.clear();
		}

		persistTrainVector(samples, results, first);
		first = false;

		samples.clear();
		results.clear();
	}

	ofs.close();
}

void dnnClassifier::extractFeatures(Place &parkingPlace, std::vector<dlib::matrix<float>>  &features) {
	// Blocks
	int step = 2; // overlap
	const cv::Size blockCount(8, 8);
	int placeWidth = (parkingPlace.frame.cols / blockCount.width), placeHeight = (parkingPlace.frame.rows / blockCount.height);
	int placeXStep = placeWidth / step, placeYStep = placeWidth / step;
	dlib::matrix<float> matrix;
	for (int y = 0; y + placeHeight < parkingPlace.frame.rows; y += placeYStep) {
		for (int x = 0; x + placeHeight < parkingPlace.frame.cols; x += placeXStep) {
			cv::Mat placeBlock = parkingPlace.frame(cv::Rect(x, y, placeWidth, placeHeight));

			// False positive reduction
			if (parkingPlace.x1 == 149 && parkingPlace.y1 == 699 && x >= 60 && x <= 110 && y >= 60 && y <= 110) {
				placeBlock = parkingPlace.frame(cv::Rect(60 - placeWidth, 60 - placeHeight, placeWidth, placeHeight));
			}
			//features.push_back(featureSobel(placeBlock, 50));
			matrix = featureCanny(placeBlock, 10, 90);
			features.push_back(matrix);
		}
	}

	// Features for whole image
	//    featureColorGradient(parkingPlace.frame, features);
	//featureHOG(parkingPlace.frame, features);
}

float dnnClassifier::featureCanny(cv::Mat &place, uchar t1, uchar t2) {
	cv::Mat srcGray, edges;

	cv::cvtColor(place, srcGray, CV_BGR2GRAY);
	cv::medianBlur(srcGray, srcGray, 9);

	cv::Canny(srcGray, edges, t1, t2);

	unsigned char n = 0;
	for (int y = 0; y < edges.rows; y++)
	{
		for (int x = 0; x < edges.cols; x++)
		{
			if (edges.at<uchar>(y, x) > 0)
			{
				n++;
			}
		}
	}

	return n;
}

void dnnClassifier::persistTrainVector(std::vector<std::vector<float>> &samples, std::vector<float> &results, bool clear) {
	std::ofstream ofs;

	if (clear) {
		ofs.open(trainVectorPath);
	}
	else {
		ofs.open(trainVectorPath, std::ofstream::app);
	}

	assert(ofs.is_open());
	for (uint i = 0; i < results.size(); i++) {
		for (auto &sample : samples[i]) {
			ofs << sample << ",";
		}

		ofs << results[i] << std::endl;
	}

	ofs.close();
}