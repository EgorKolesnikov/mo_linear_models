#pragma once

#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>

#include "reader.h"
#include "util.h"

using std::vector;
using std::exception;
using std::ofstream;
using std::ifstream;
using std::fixed;
using std::showpoint;
using std::setprecision;


class AbstractLinearModel {

protected:
	// Model params
	int _n_features;
	int _n_iterations;
	double _learning_rate;
	double _learning_rate_decay;

	// Main hyperplain params
	vector<double> w_prev;
	vector<double> w_cur;

	// Model specific
	virtual void _update_w(vector<DatasetEntry>& batch) = 0;

	virtual double _evaluate_one(DatasetEntry& entry);
	virtual void _update_state();
	virtual bool _stop_rule();
	virtual bool _run_iteration(vector<DatasetEntry>& batch);

public:
	AbstractLinearModel(
		int n_features
		, int _n_iterations
		, double _learning_rate
		, double _learning_rate_decay
	);

	void save(const string& path);
	void load(const string& path);
	vector<double> get_hyperplane();

	virtual void fit(TFullDataReader&);
	
	virtual double predict_one(DatasetEntry&) = 0;
	virtual vector<double> predict(vector<DatasetEntry>&);
	virtual double evaluate(TFullDataReader&) = 0;
};
