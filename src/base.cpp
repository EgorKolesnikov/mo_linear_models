#include "base.h"


TAbstractLinearModel::TAbstractLinearModel(
	int n_features = 0
	, int n_iterations = 5
	, double learning_rate = 0.01
	, double learning_rate_decay = 1.0
)
	: _n_features(n_features)
	, _n_iterations(n_iterations)
	, _learning_rate(learning_rate)
	, _learning_rate_decay(learning_rate_decay)

	// +1 for b in 'wx + b'
	, w_prev(n_features + 1, 0.0)
	, w_cur(n_features + 1, 0.0)
{ }

void TAbstractLinearModel::_update_state(){
	this->_learning_rate /= this->_learning_rate_decay;
}

bool TAbstractLinearModel::_stop_rule(){
	return false;
}

bool TAbstractLinearModel::_run_iteration(vector<DatasetEntry>& batch){
	this->_update_rule(batch);

	if(this->_stop_rule()){
		return false;
	}

	this->_update_state();
	return true;
}

void TAbstractLinearModel::save(const string& path){
	
}

void TAbstractLinearModel::load(const string& path){
	
}

vector<double> TAbstractLinearModel::get_hyperplane(){
	return this->w_cur;
}

void TAbstractLinearModel::fit(TFullDataReader& reader){
	for(int i = 0; i < this->_n_iterations; ++i){
		vector<DatasetEntry> batch = reader.next_batch();
		if(!this->_run_iteration(batch)){
			break;
		}
	}
}
