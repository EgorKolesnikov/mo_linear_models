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
{ }


void SgdClassification::_update_rule(vector<DatasetEntry>& batch){
	/*
	*	self.w = self.w - self.lr * self._batch_grad(batch);
	*/
	_copy(this->w_cur, this->w_prev);
	
	vector<double> grad = batch_grad(this->w_cur, batch);

	_mul(grad, this->_learning_rate);
	_sub(this->w_cur, grad);
}
