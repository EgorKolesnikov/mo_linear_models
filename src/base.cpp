#include "base.h"


TAbstractLinearModel::TAbstractLinearModel(int n_features = 0, int n_iterations = 5, double learning_rate = 0.01)
	: _n_features(n_features)
	, _n_iterations(n_iterations)
	, _learning_rate(learning_rate)

	// +1 for b in 'wx + b'
	, w(n_features + 1, 0.0)
{ }


void TAbstractLinearModel::save(){
	
}

void TAbstractLinearModel::load(){
	
}

void TAbstractLinearModel::train(TFullDataReader& reader){
	throw exception();
}

void TAbstractLinearModel::predict_one(){
	throw exception();
}

void TAbstractLinearModel::predict(){
	throw exception();
}
