#pragma once

#include <vector>
#include <exception>

#include "reader.h"
#include "util.h"

using std::vector;
using std::exception;


class TAbstractLinearModel {

protected:
	// Model params
	int _n_features;
	int _n_iterations;
	double _learning_rate;
	double _learning_rate_decay;

	// Main hyperplain params
	vector<double> w_prev;
	vector<double> w_cur;

	virtual void _update_rule(vector<DatasetEntry>& batch) = 0;
	virtual void _update_state();
	virtual bool _stop_rule();
	virtual bool _run_iteration(vector<DatasetEntry>& batch);

public:
	TAbstractLinearModel(
		int n_features
		, int _n_iterations
		, double _learning_rate
		, double _learning_rate_decay
	);

	void save(const string& path);
	void load(const string& path);
	vector<double> get_hyperplane();

	virtual void fit(TFullDataReader&);
};
