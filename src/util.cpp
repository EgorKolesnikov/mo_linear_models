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

void _dot(const vector<double>& v1, const vector<double>& v2, vector<double>& to_store){
	for(size_t i = 0; i < v1.size(); ++i){
		to_store[i] = v1[i] * v2[i];
	}
}

double _sum(const vector<double>& v){
	double result = 0.0;
	for(size_t i = 0; i < v.size(); ++i){
		result += v[i];
	}
	return result;
}

double _accuracy(const vector<double>& pred, const vector<double>& labels){
	int correct = 0;
	for(size_t i = 0; i < pred.size(); ++i){
		if(int(pred[i]) == int(labels[i])){
			correct += 1;
		}
	}
	return 100.0 * correct / pred.size();
}
