#include <iostream>
#include "objdetect/Classifier.h"

int main() {	
	Classifier classifier("strecha1_map.txt", "train_vector.txt", "svm.xml");
	classifier.train("full.txt", "free.txt");
	classifier.recognize("test.txt", "results.txt");
	classifier.evaluate("groundtruth.txt", "results.txt");
    return 0;
}