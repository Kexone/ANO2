#ifndef TESTSVM_H
#define TESTSVM_H

#include "DifferentialEvolution.h"
#include "objdetect/Classifier.h"


class TestSvm : public de::IOptimizable
{
private:
	double EvaluteCost(std::vector<double> inputs) const override;
	unsigned int NumberOfParameters() const override;
	std::vector<Constraints> GetConstraints() const override;

	unsigned int m_dim;

public:
	TestSvm(unsigned int dims)
	{
		m_dim = dims;
	}
};

#endif // TESTSVM_H