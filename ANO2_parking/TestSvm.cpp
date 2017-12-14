#include "testsvm.h"

double TestSvm::EvaluteCost(std::vector<double> inputs) const
{
	Classifier classifier("strecha1_map.txt", "train_vector.txt", "svm.xml",0);
	classifier.train("full.txt", "free.txt",inputs[0],inputs[1],inputs[2]* 100);
	return 1 - classifier.evaluate("groundtruth.txt", "results.txt");
}

unsigned int TestSvm::NumberOfParameters() const
{
	return m_dim;
}

std::vector<de::IOptimizable::Constraints> TestSvm::GetConstraints() const
{
	std::vector<Constraints> constr;

	constr.push_back(Constraints(0.0, 2.0, true));
	constr.push_back(Constraints(0.0, 1.0, true));
	constr.push_back(Constraints(1.0, 5.0, true));
	//constr.push_back(Constraints(0.3, 1.0, true));
	return constr;
}