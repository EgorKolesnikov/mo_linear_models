#include "sgd.h"
#include "reader.h"
#include "util.h"


AbstractSgdModel::AbstractSgdModel(
	int n_features = 0
	, int n_iterations = 5
	, double learning_rate = 0.01
	, double learning_rate_decay = 1.0
)
	: AbstractLinearModel(n_features, n_iterations, learning_rate, learning_rate_decay)
{ }

double AbstractSgdModel::_predict_one(DatasetEntry& entry){
	return AbstractSgdModel::_evaluate_one(entry);
}

double AbstractSgdModel::predict_one(DatasetEntry& entry){
	return this->_predict_one(entry);
}

void AbstractSgdModel::_update_w_one_direct(DatasetEntry& entry){
	double grad = this->_gradient_one(entry);
	for(int i = 0; i < this->_n_features; ++i){
		this->w_cur[i] = this->w_prev[i] + this->_learning_rate * grad * entry.x[i];
	}
	this->w_cur[this->_n_features] = this->w_prev[this->_n_features] + this->_learning_rate * grad * 1.0;
}

void AbstractSgdModel::_update_w_one_cache(DatasetEntry& entry, int idx_in_batch){
	double grad = this->_gradient_one(entry);
	for(int i = 0; i < this->_n_features; ++i){
		this->parallel_batch_entries[idx_in_batch][i] = this->_learning_rate * grad * entry.x[i];
	}
	this->parallel_batch_entries[idx_in_batch][this->_n_features] = this->_learning_rate * grad * 1.0;
}


/*
*	Logistic regression
*/

LogisticRegressionModel::LogisticRegressionModel(
	int n_features = 0
	, int n_iterations = 5
	, double learning_rate = 0.01
	, double learning_rate_decay = 1.0
)
	: AbstractSgdModel(n_features, n_iterations, learning_rate, learning_rate_decay)
{ }

double LogisticRegressionModel::_predict_one(DatasetEntry& entry){
	double pred = AbstractLinearModel::_evaluate_one(entry);
	return 1.0 / (1.0 + exp(-pred));
}

double LogisticRegressionModel::_gradient_one(DatasetEntry& entry){
	double pred = this->_predict_one(entry);
	double error = entry.y - pred;
	return error;
}

double LogisticRegressionModel::predict_one(DatasetEntry& entry){
	double val = this->_predict_one(entry);
	return round(val);
}

double LogisticRegressionModel::evaluate(TFullDataReader& reader){
	int correct = 0;

    vector<DatasetEntry> batch;
    long dataset_size = 0;
    while ((batch = reader.next_batch()).size() > 0) {
        dataset_size += batch.size();
        for (DatasetEntry &entry : batch) {
            double prediction = this->predict_one(entry);
            if (prediction == entry.y) {
                correct += 1;
            }
        }
    }

	return 100.0 * correct / dataset_size;
}


/*
*	Linear regression
*/

LinearRegressionModel::LinearRegressionModel(
	int n_features = 0
	, int n_iterations = 5
	, double learning_rate = 0.01
	, double learning_rate_decay = 1.0
)
	: AbstractSgdModel(n_features, n_iterations, learning_rate, learning_rate_decay)
{ }

double LinearRegressionModel::_gradient_one(DatasetEntry& entry){
	double pred = this->_predict_one(entry);
	double error = (pred - entry.y);
	double grad = error;
	return -grad;
}

double LinearRegressionModel::evaluate(TFullDataReader& reader){
	double mse = 0;


    vector<DatasetEntry> batch;
	while ((batch = reader.next_batch()).size() > 0) {
        for (DatasetEntry &entry : batch) {
            double prediction = this->predict_one(entry);
            double error = (prediction - entry.y);
            double loss = error * error;
            mse += loss;
        }
    }

	return mse;
}
