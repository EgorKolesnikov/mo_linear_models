#pragma once

#include "base.h"


class SgdClassification : public TAbstractLinearModel {

protected:
	void _gradient(DatasetEntry& entry, vector<double>& to_store);
	void _gradient_batch(vector<DatasetEntry>& batch, vector<double>& to_store);
	void _update_rule(vector<DatasetEntry>& batch);
	double _predict_one(DatasetEntry& entry);

public:
	vector<double> temp;

	SgdClassification(
		int n_features
		, int _n_iterations
		, double _learning_rate
		, double learning_rate_decay
	);

	double evaluate(vector<DatasetEntry>&);
};
