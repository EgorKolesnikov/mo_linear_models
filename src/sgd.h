#pragma once

#include <math.h>
#include <cmath>
#include <omp.h>

#include "base.h"


class AbstractSgdModel : public AbstractLinearModel {

protected:
	// Model specific
	virtual double _gradient_one(DatasetEntry& entry) = 0;
	virtual double _predict_one(DatasetEntry& entry);
	
	virtual void _update_w_one_direct(DatasetEntry& entry);
	virtual void _update_w_one_cache(DatasetEntry& entry, int idx_in_batch);

public:
	AbstractSgdModel(
		int n_features
		, int _n_iterations
		, double _learning_rate
		, double learning_rate_decay
	);

	virtual double predict_one(DatasetEntry&);
	virtual double evaluate(TFullDataReader&) = 0;
};


class LogisticRegressionModel : public AbstractSgdModel {

protected:
	virtual double _predict_one(DatasetEntry& entry);
	virtual double _gradient_one(DatasetEntry& entry);

public:
	LogisticRegressionModel(
		int n_features
		, int _n_iterations
		, double _learning_rate
		, double learning_rate_decay
	);

	virtual double predict_one(DatasetEntry&);
	double evaluate(TFullDataReader&);
};


class LinearRegressionModel : public AbstractSgdModel {

protected:
	virtual double _gradient_one(DatasetEntry& entry);

public:
	LinearRegressionModel(
		int n_features
		, int _n_iterations
		, double _learning_rate
		, double learning_rate_decay
	);

	double evaluate(TFullDataReader&);
};
