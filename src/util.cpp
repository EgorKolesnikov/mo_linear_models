#include "util.h"


void _add(vector<double>& a, const vector<double>& b){
	for(size_t i = 0; i < a.size(); ++i){
		a[i] += b[i];
	}
}

void _sub(vector<double>& a, const vector<double>& b){
	for(size_t i = 0; i < a.size(); ++i){
		a[i] -= b[i];
	}
}

void _mul(vector<double>& a, int value){
	for(size_t i = 0; i < a.size(); ++i){
		a[i] *= value;
	}
}

void _div(vector<double>& a, int value){
	for(size_t i = 0; i < a.size(); ++i){
		a[i] /= value;
	}
}

void _copy(const vector<double>& src, vector<double>& dst){
	for(size_t i = 0; i < src.size(); ++i){
		dst[i] = src[i];
	}
}

vector<double> grad(vector<double>& w, DatasetEntry& x, double reg){
	/*
	*	-y * x * (1. - 1. / (1. + exp(-y * dot(w, x)))) + reg * w
	*/
	vector<double> result(w.size(), 0.0);

	// TODO

	return result;
}

vector<double> batch_grad(vector<double>& w, vector<DatasetEntry>& batch, double reg){
	/*
	*	mean([grad(w, x, reg) for x in batch], axis=0)
	*/
	vector<double> result(w.size(), 0.0);

	for(DatasetEntry& x : batch){
		_add(result, grad(w, x, reg));
	}

	_div(result, static_cast<int>(batch.size()));
	return result;
}
