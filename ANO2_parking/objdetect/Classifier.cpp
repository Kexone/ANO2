#include "Classifier.h"
#include <opencv2/ml/ml.hpp>
#include "../utils/Loader.h"
#include <opencv2/opencv.hpp>
#include "omp.h"


#define mSVM 0
#define mNEURAL 1


void Classifier::drawParkingPlaces(cv::Mat &scene, std::vector<Place> &parkingPlaces) {
	int occupied = 0;

	for (auto &place : parkingPlaces) {
		cv::Scalar color = (place.occupied) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
		occupied = (place.occupied) ? occupied + 1 : occupied;

		cv::line(scene, cv::Point(place.x1, place.y1), cv::Point(place.x2, place.y2), color);
		cv::line(scene, cv::Point(place.x2, place.y2), cv::Point(place.x3, place.y3), color);
		cv::line(scene, cv::Point(place.x3, place.y3), cv::Point(place.x4, place.y4), color);

		cv::Point center((place.x1 + place.x2 + place.x3 + place.x4) / 4, (place.y1 + place.y2 + place.y3 + place.y4) / 4);
		cv::circle(scene, center, 6, color, -1);
	}

	cv::rectangle(scene, cv::Point(0, 0), cv::Point(130, 55), cv::Scalar(0, 0, 0), -1);
	std::stringstream ss;
	ss << "Occupied: " << occupied;
	cv::putText(scene, ss.str(), cv::Point(10, 20), CV_FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1, CV_AA);
	ss.str("");
	ss << "Free: " << parkingPlaces.size() - occupied;
	cv::putText(scene, ss.str(), cv::Point(10, 40), CV_FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1, CV_AA);
}

void Classifier::featureHOG(const cv::Mat &place, std::vector<float> &features) {
	std::vector<float> descriptors;
	cv::HOGDescriptor hog;
	hog.winSize = place.size();
	hog.blockSize = cv::Size(128, 128);
	hog.cellSize = cv::Size(64, 64);
	//cv::HOGDescriptor hog(place.size(),cv::Size(128,128),cv::Size(64,64), cv::Size(64, 64), 9);
	hog.compute(place, descriptors);

	for (auto &d : descriptors) {
		features.emplace_back(d);
	}
}

float Classifier::featureSobel(cv::Mat &place, int minThreshold) {
	cv::Mat src = place.clone();
	cv::Mat srcGray, gradX, gradY, grad;

	cv::cvtColor(src, srcGray, CV_BGR2GRAY);
	cv::medianBlur(srcGray, srcGray, 3);

	Sobel(srcGray, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	Sobel(srcGray, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

	convertScaleAbs(gradX, gradX);
	convertScaleAbs(gradY, gradY);

	addWeighted(gradX, 0.5, gradY, 0.5, 0, grad);

	float edges = 0;
	for (int y = 0; y < grad.rows; y++) {
		for (int x = 0; x < grad.cols; x++) {
			if (grad.at<uchar>(y, x) > minThreshold) {
				edges++;
			}
		}
	}

	return edges;
}

float Classifier::featureCanny(cv::Mat &place, uchar t1, uchar t2) {
	cv::Mat srcGray, edges;

	cv::cvtColor(place, srcGray, CV_BGR2GRAY);
	cv::medianBlur(srcGray, srcGray, 9);

	cv::Canny(srcGray, edges, t1, t2);

	float n = 0;
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

void Classifier::extractFeatures(Place &parkingPlace, std::vector<float> &features) {
	if (typeClass == mSVM) {
		int step = 2; // overlap
		const cv::Size blockCount(8, 8);
		int placeWidth = (parkingPlace.frame.cols / blockCount.width), placeHeight = (parkingPlace.frame.rows / blockCount.height);
		int placeXStep = placeWidth / step, placeYStep = placeWidth / step;

		for (int y = 0; y + placeHeight < parkingPlace.frame.rows; y += placeYStep) {
			for (int x = 0; x + placeHeight < parkingPlace.frame.cols; x += placeXStep) {
				cv::Mat placeBlock = parkingPlace.frame(cv::Rect(x, y, placeWidth, placeHeight));

				if (parkingPlace.x1 == 149 && parkingPlace.y1 == 699 && x >= 60 && x <= 110 && y >= 60 && y <= 110) {
					placeBlock = parkingPlace.frame(cv::Rect(60 - placeWidth, 60 - placeHeight, placeWidth, placeHeight));
				}
				//features.push_back(featureSobel(placeBlock, 50));
				features.push_back(featureCanny(placeBlock, 10, 90));
			}
		}

		featureHOG(parkingPlace.frame, features);
	}
	else if(typeClass == mNEURAL)
	{
		cv::Mat edges, srcGray = parkingPlace.frame.clone();
		cv::resize(srcGray, srcGray, cv::Size(40,40));

		cv::Canny(srcGray, edges, 10, 90);

		for (int y = 0; y < edges.rows; y++) {
			for (int x = 0; x < edges.cols; x++) {
				features.push_back(static_cast<float>(edges.at<uchar>(y, x)));
			}
		}
	}
}

void Classifier::persistTrainVector(std::vector<std::vector<float>> &samples, std::vector<float> &results, bool clear) {
	std::ofstream ofs;

	if (clear) {
		ofs.open(trainVectorPath);
	}
	else {
		ofs.open(trainVectorPath, std::ofstream::app);
	}

	assert(ofs.is_open());
	assert(samples.size() == results.size());

	for (int i = 0; i < results.size(); i++) {
		for (auto &sample : samples[i]) {
			ofs << sample << ",";
		}

		ofs << results[i] << std::endl;
	}

	ofs.close();
}

void Classifier::generateTrainVector(std::string &filesListPath, float expectedResult, bool clear) {
	std::ifstream ofs;
	ofs.open(filesListPath);
	assert(ofs.is_open());

	bool first = clear; // On first pass, clean the file

	while (!ofs.eof()) {
		std::string line;
		ofs >> line;

		std::cout << "Loading: " << line << std::endl;
		cv::Mat scene = cv::imread(line, CV_LOAD_IMAGE_COLOR);

		// Load places geometry
		std::vector<Place> places;
		Loader::loadParkingPlaces(geometryDataPath, scene, places);

		std::vector<std::vector<float>> samples(places.size());
		std::vector<float> results(places.size()), features(places.size());
//#pragma omp parallel for shared(extractFeatures,samples, results)
		for(uint i = 0; i < places.size(); i++) {
		//for (auto &parkingPlace : places) {
			extractFeatures(places[i], features);
			samples[i] = features;
			results[i] = expectedResult;
			features.clear();
		}

		persistTrainVector(samples, results, first);
		first = false;

		samples.clear();
		results.clear();
	}

	ofs.close();
}

void Classifier::loadTrainVector(cv::Mat &samples, cv::Mat &responses) {

	std::vector<std::vector<float>> descriptor;
	std::vector<float> classes;

	std::ifstream ifs;
	ifs.open(trainVectorPath);
	assert(ifs.is_open());

	while (!ifs.eof()) {
		std::string str;
		ifs >> str;

		std::vector<float> desc;
		size_t found = str.find(',');

		while (found != std::string::npos) {
			std::string t = str.substr(0, found);
			float d = std::stof(t);
			float nearest = floorf(d * 1000.0f + 0.5f) / 1000.0f;
			desc.push_back(nearest);

			// Finding other descriptor
			str = str.substr(found + 1);
			found = str.find(',');
		}

		if (!desc.empty()) {
			descriptor.push_back(desc);
			classes.push_back(std::stof(str));
		}
	}

	ifs.close();

	auto numSamples = static_cast<int>(descriptor.size());
	auto numFeatures = static_cast<int>(descriptor[0].size());

	samples = cv::Mat(numSamples, numFeatures, CV_32F);
	if (typeClass == mSVM)
		responses = cv::Mat(static_cast<int>(descriptor.size()), 1, CV_32S);
	if(typeClass == mNEURAL)
		responses = cv::Mat(numSamples, 2, CV_32F);

	for (int i = 0; i < numSamples; i++) {
		for (int j = 0; j < numFeatures; j++) {
			samples.at<float>(i, j) = descriptor[i][j];
			if (typeClass == mSVM)
			{
				responses.at<int>(i, 0) = static_cast<int>(classes[i]);
			}
			else if(typeClass == mNEURAL)
			{
				responses.at<float>(i, 0) = 0;
				responses.at<float>(i, 0) = 0;
				responses.at<float>(i, static_cast<int>(classes[i])) = 1;
			}
		}
	}
}

void Classifier::train(std::string posFilesListPath, std::string negFilesListPath, double c, double nu, double gamma) {
	generateTrainVector(posFilesListPath, 1.0f, true);
	if (typeClass == mSVM)
		generateTrainVector(negFilesListPath, -1.0f);
	if (typeClass == mNEURAL)
	generateTrainVector(negFilesListPath, 0.0f);

	std::cout << "Initializing training..." << std::endl;

	cv::Mat samples, responses;
	loadTrainVector(samples, responses);
	if (typeClass == mSVM) {

		cv::TermCriteria criteria;
		criteria.type = CV_TERMCRIT_EPS;
		criteria.epsilon = 1.0;
		criteria.maxCount = gamma;

		std::cout << "Initializing SVM..." << std::endl;

		svm = cv::ml::SVM::create();
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::INTER);
		svm->setGamma(0.1);
		svm->setC(c);
		svm->setNu(nu);
		svm->setTermCriteria(criteria);

		std::cout << "Training SVM...";
		clock_t interval = cv::getTickCount();
		if (!svm->train(samples, cv::ml::ROW_SAMPLE, responses)) {
			std::cout << "ERROR: training failed" << std::endl;
		}
		else {
			svm->save(svmTrainedDataPath);
			std::ofstream file;
			file.open("resultTrain.txt", std::ios::app);
			file << "par gamma: " << gamma << std::endl;
			file << "par C: " << c << std::endl;
			file << "par NU: " << nu << std::endl << std::endl << std::endl;
			std::cout << "FINISHED" << ((double)cv::getTickCount() - interval) / cv::getTickFrequency() / 60.0 << std::endl;
		}
	}
	else if(typeClass == mNEURAL)
	{
		cv::Mat layerSize = cv::Mat(3,1,CV_32S);
		layerSize.at<int>(0, 0) = samples.cols; 
		layerSize.at<int>(1, 0) = samples.cols/2; 
		layerSize.at<int>(2, 0) = responses.cols; 
		float epsilon = 0.0001;
		int maxCount = 150;

		machineLearn = cv::ml::ANN_MLP::create();
		machineLearn->setLayerSizes(layerSize);
		machineLearn->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
		machineLearn->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, epsilon);
		machineLearn->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, maxCount, epsilon));
		std::cout << "Training neutral network...";
		clock_t interval = cv::getTickCount();
		if (!machineLearn->train(samples, cv::ml::ROW_SAMPLE, responses)) {
			std::cout << "ERROR: training failed" << std::endl;
		}
		else
		{
			machineLearn->save(svmTrainedDataPath);
			std::cout << "FINISHED" << ((double)cv::getTickCount() - interval) / cv::getTickFrequency() / 60.0 << std::endl;
		}

	}
}

void Classifier::recognize(std::string testFilesListPath, std::string resultsFilePath) {
	if(typeClass == mSVM)
		svm = cv::ml::SVM::load(svmTrainedDataPath);
	if (typeClass == mNEURAL)
		machineLearn = cv::ml::ANN_MLP::load(svmTrainedDataPath);
	std::vector<Place> places;
	std::vector<float> features;

	std::ifstream ifs;
	ifs.open(testFilesListPath);
	assert(ifs.is_open());

	std::ofstream resultsStream;
	resultsStream.open(resultsFilePath);
	assert(resultsStream.is_open());

	while (!ifs.eof()) {
		int occupied = 0;
		std::string line;
		ifs >> line;

		cv::Mat scene = cv::imread(line, CV_LOAD_IMAGE_COLOR);

		Loader::loadParkingPlaces(geometryDataPath, scene, places);

		for (auto &place : places) {
			extractFeatures(place, features);
			auto numFeatures = static_cast<int>(features.size());
			cv::Mat resMat = cv::Mat(1, numFeatures, CV_32FC1);

			for (int i = 0; i < numFeatures; i++) {
				resMat.at<float>(0, i) = features[i];
			}

			float response;
			if (typeClass == mSVM)
				response = svm->predict(resMat);
			if (typeClass == mNEURAL)
				response = machineLearn->predict(resMat);
			place.occupied = response > 0;
			occupied += place.occupied;

			resultsStream << place.occupied << std::endl;
			features.clear();
		}

		// Show results
		//        cv::Mat results = scene.clone();
		//        drawParkingPlaces(results, places);
		//       cv::namedWindow("Results", CV_WINDOW_NORMAL);
		//        cv::imshow("Results", results);
		//        cv::waitKey(0);

		places.clear();
	}

	ifs.close();
	resultsStream.close();
}

float Classifier::evaluate(std::string groundTruthFile, std::string resultsFilePath) {
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

		if ((detect == 1) && (ground == 0)) {
			falsePositives++;
		}

		if ((detect == 0) && (ground == 1)) {
			falseNegatives++;
		}

		if ((detect == 1) && (ground == 1)) {
			truePositives++;
		}

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
	return acc;
}

const std::string &Classifier::getGeometryDataPath() const {
	return geometryDataPath;
}

void Classifier::setGeometryDataPath(const std::string &geometryDataPath) {
	Classifier::geometryDataPath = geometryDataPath;
}

const std::string &Classifier::getTrainVectorPath() const {
	return trainVectorPath;
}

void Classifier::setTrainVectorPath(const std::string &trainVectorPath) {
	Classifier::trainVectorPath = trainVectorPath;
}

const std::string &Classifier::getSvmTrainedDataPath() const {
	return svmTrainedDataPath;
}

void Classifier::setSvmTrainedDataPath(const std::string &svmTrainedDataPath) {
	Classifier::svmTrainedDataPath = svmTrainedDataPath;
}