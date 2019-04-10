#include "sgd.h"
#include "reader.h"
#include "util.h"


SgdClassification::SgdClassification(
	int n_features = 0
	, int n_iterations = 5
	, double learning_rate = 0.01
	, double learning_rate_decay = 1.0
)
	: TAbstractLinearModel(n_features, n_iterations, learning_rate, learning_rate_decay)
	, temp(_n_features + 1, 0.0)
{ }


void SgdClassification::_gradient(DatasetEntry& entry, vector<double>& to_store){

}


void SgdClassification::_gradient_batch(vector<DatasetEntry>& batch, vector<double>& to_store){
	
}


void SgdClassification::_update_rule(vector<DatasetEntry>& batch){
	/*
	*	self.w = self.w - self.lr * self._batch_grad(batch);
	*/

	// save w_cur W to w_prev
	_copy(this->w_cur, this->w_prev);
	
	// evaluate gradient
	std::fill(this->temp.begin(), this->temp.end(), 0.0);
	this->_gradient_batch(batch, this->temp);

	// apply w -= lr * grad
	_mul(this->temp, this->_learning_rate);
	_sub(this->w_cur, this->temp);
}


double SgdClassification::_predict_one(DatasetEntry& batch){
	vector<double> t(batch.x.size());
	_dot(batch.x, this->get_hyperplane(), t);
	double v = _sum(t);
	return v > 0 ? 1.0 : 0.0;
}

double SgdClassification::evaluate(vector<DatasetEntry>& dataset){
	vector<double> labels;
	labels.reserve(dataset.size());
	for(DatasetEntry& e : dataset){
		labels.push_back(e.y);
	}

	vector<double> predictions = this->predict(dataset);
	return _accuracy(predictions, labels);
}