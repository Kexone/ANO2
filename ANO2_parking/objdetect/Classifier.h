#ifndef VSB_ANO2_PARKING_CLASSIFIER_H
#define VSB_ANO2_PARKING_CLASSIFIER_H

#include <utility>
#include <opencv2/ml/ml.hpp>



struct Place;

class Classifier {
private:
    cv::Ptr< cv::ml::SVM > svm;
	cv::Ptr< cv::ml::ANN_MLP > machineLearn;
    std::string geometryDataPath;
    std::string trainVectorPath;
    std::string svmTrainedDataPath;

    float featureSobel( cv::Mat &place, int minThreshold);
	float featureCanny(cv::Mat &place, uchar t1, uchar t2);
    void extractFeatures(Place &parkingPlace, std::vector<float> &features);
    void generateTrainVector(std::string &filesListPath, float expectedResult, bool clear = false);
    void persistTrainVector(std::vector<std::vector<float>> &samples, std::vector<float> &results, bool clear = false);
    void loadTrainVector(cv::Mat &samples, cv::Mat &responses);
	void featureHOG(const cv::Mat &place, std::vector<float> &features);
	int typeClass;
public:
    Classifier(std::string geometryDataPath, std::string trainVectorPath, std::string svmTrainedDataPath, int type)
            : geometryDataPath(std::move(geometryDataPath)), trainVectorPath(std::move(trainVectorPath)), svmTrainedDataPath(
            std::move(svmTrainedDataPath)), typeClass(std::move(type)) {}
	
    static void drawParkingPlaces(cv::Mat &scene, std::vector<Place> &parkingPlaces);
    void train(std::string posFilesListPath, std::string negFilesListPath, double c = 2.0, double nu = 0.1, double gamma = 0.1);
    void recognize(std::string testFilesListPath, std::string resultsFilePath);
	float evaluate(std::string groundTruthFile, std::string resultsFilePath);

    const std::string &getGeometryDataPath() const;
    const std::string &getTrainVectorPath() const;
    const std::string &getSvmTrainedDataPath() const;

    void setGeometryDataPath(const std::string &geometryDataPath);
    void setTrainVectorPath(const std::string &trainVectorPath);
    void setSvmTrainedDataPath(const std::string &svmTrainedDataPath);
};

#endif //VSB_ANO2_PARKING_CLASSIFIER_H
