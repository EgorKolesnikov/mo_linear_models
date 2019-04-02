#pragma once

#include <vector>
#include <exception>

#include "reader.h"

using std::vector;
using std::exception;


class TAbstractLinearModel {

protected:
	// Model params
	int _n_features;
	int _n_iterations;
	double _learning_rate;

	// Main hyperplain params
	vector<double> w;

public:
	TAbstractLinearModel(
		int n_features = 0
		, int _n_iterations = 5
		, double _learning_rate = 0.1
	);

	void save();
	void load();

	virtual void train(TFullDataReader&);
	virtual void predict_one();
	virtual void predict();
};
