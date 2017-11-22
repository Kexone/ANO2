#ifndef VSB_ANO2_PARKING_CLASSIFIER_H
#define VSB_ANO2_PARKING_CLASSIFIER_H

#include <utility>
#include <opencv2/ml/ml.hpp>
struct Place;

class Classifier {
private:
    cv::Ptr<cv::ml::SVM> svm;
    std::string geometryDataPath;
    std::string trainVectorPath;
    std::string svmTrainedDataPath;

    float featureSobel(const cv::Mat &place, int minThreshold);

    void extractFeatures(Place &parkingPlace, std::vector<float> &features);
    void generateTrainVector(std::string &filesListPath, float expectedResult, bool clear = false);
    void persistTrainVector(std::vector<std::vector<float>> &samples, std::vector<float> &results, bool clear = false);
    void loadTrainVector(cv::Mat &samples, cv::Mat &responses);
	void featureColorGradient(const cv::Mat &place, std::vector<float> &features);
	void featureHOG(const cv::Mat &place, std::vector<float> &features);
public:
    Classifier(std::string geometryDataPath, std::string trainVectorPath, std::string svmTrainedDataPath)
            : geometryDataPath(std::move(geometryDataPath)), trainVectorPath(std::move(trainVectorPath)), svmTrainedDataPath(
            std::move(svmTrainedDataPath)) {}

    static void drawParkingPlaces(cv::Mat &scene, std::vector<Place> &parkingPlaces);
    void train(std::string posFilesListPath, std::string negFilesListPath);
    void recognize(std::string testFilesListPath, std::string resultsFilePath);
	void evaluate(std::string groundTruthFile, std::string resultsFilePath);

    const std::string &getGeometryDataPath() const;
    const std::string &getTrainVectorPath() const;
    const std::string &getSvmTrainedDataPath() const;

    void setGeometryDataPath(const std::string &geometryDataPath);
    void setTrainVectorPath(const std::string &trainVectorPath);
    void setSvmTrainedDataPath(const std::string &svmTrainedDataPath);
};

#endif //VSB_ANO2_PARKING_CLASSIFIER_H
