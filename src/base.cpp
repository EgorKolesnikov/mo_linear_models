#include "base.h"


AbstractLinearModel::AbstractLinearModel(
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

double AbstractLinearModel::_evaluate_one(DatasetEntry& entry){
	double pred = 0.0;
	for(int i = 0; i < this->_n_features; ++i){
		pred += entry.x[i] * this->w_prev[i];
	}
	pred += 1.0 * this->w_prev.back();
	return pred;
}

void AbstractLinearModel::_update_state(){
	this->_learning_rate /= this->_learning_rate_decay;
}

bool AbstractLinearModel::_stop_rule(){
	return false;
}

bool AbstractLinearModel::_run_iteration(vector<DatasetEntry>& batch){
	this->_update_w(batch);

	if(this->_stop_rule()){
		return false;
	}

	this->_update_state();
	return true;
}

void AbstractLinearModel::save(const string& path){
	ofstream outf(path);
	outf << fixed << showpoint << setprecision(5);

	vector<double> hyperplane = this->get_hyperplane();
	for(int i = 0; i < this->_n_features + 1; ++i){
		outf << hyperplane[i] << " ";
	}

	outf.close();
}

void AbstractLinearModel::load(const string& path){
	ifstream inf(path);

	this->w_prev = vector<double>(this->_n_features + 1, 0.0);
	this->w_cur = vector<double>(this->_n_features + 1, 0.0);

	for(int i = 0; i < this->_n_features + 1; ++i){
		double w;
		inf >> w;

		this->w_prev[i] = w;
		this->w_cur[i] = w;
	}

	inf.close();
}

vector<double> AbstractLinearModel::get_hyperplane(){
	return this->w_cur;
}

void AbstractLinearModel::fit(TFullDataReader& reader){
	for(int i = 0; i < this->_n_iterations; ++i){
		// printf("Iteration %d\n", i);

		vector<DatasetEntry> batch = reader.next_batch();
		if(!this->_run_iteration(batch)){
			break;
		}
	}
}

vector<double> AbstractLinearModel::predict(vector<DatasetEntry>& dataset){
	vector<double> predictions;
	predictions.reserve(dataset.size());

	for(DatasetEntry& entry: dataset){
		predictions.push_back(this->predict_one(entry));
	}

	return predictions;
}
