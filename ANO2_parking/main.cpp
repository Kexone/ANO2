#include <iostream>
#include "objdetect/Classifier.h"
#include "testSvm.h"
int main() {	

	//TestSvm svm(3);
	//de::DifferentialEvolution de(svm, 50);
	//de.Optimize(1000, true);


	Classifier classifier("strecha1_map.txt", "train_vector.txt", "neural.xml",1);
	classifier.train("full.txt", "free.txt");
	classifier.recognize("test.txt", "results.txt");
	classifier.evaluate("groundtruth.txt", "results.txt");
    return 0;
}