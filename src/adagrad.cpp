#include "adagrad.h"
#include "reader.h"
#include "util.h"


LogisticRegressionModelForAdagrad::LogisticRegressionModelForAdagrad(
        int n_features = 0
        , int n_iterations = 5
        , double learning_rate = 0.01
        , double learning_rate_decay = 2.0
                , double eps = 0.01
)
        : LogisticRegressionModel(n_features, n_iterations, learning_rate, learning_rate_decay)
        , temp(_n_features + 1, 0.0)
        , G(_n_features + 1, 0.0)
        , eps(eps)
{ }

void LogisticRegressionModelForAdagrad::_update_w_one(DatasetEntry& entry){
    double grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->w_cur[i] = this->w_prev[i] + this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps);
    }
    G[this->_n_features] += grad * grad;
    this->w_cur[this->_n_features] = this->w_prev[this->_n_features] +
            this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}

void LogisticRegressionModelForAdagrad::_update_w(vector<DatasetEntry>& batch){
    _copy(this->w_cur, this->w_prev);
    for(DatasetEntry& entry : batch){
        this->_update_w_one(entry);
    }
}

LinearRegressionModelForAdagrad::LinearRegressionModelForAdagrad(
        int n_features = 0
        , int n_iterations = 5
        , double learning_rate = 0.01
        , double learning_rate_decay = 1.0
                , double eps = 0.01
)
        : LinearRegressionModel(n_features, n_iterations, learning_rate, learning_rate_decay)
        , temp(_n_features + 1, 0.0)
        , G(_n_features + 1, 0.0)
        , eps(eps)
{ }

void LinearRegressionModelForAdagrad::_update_w_one(DatasetEntry& entry){
    double grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->w_cur[i] = this->w_prev[i] + this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps);
    }
    G[this->_n_features] += grad * grad;
    this->w_cur[this->_n_features] = this->w_prev[this->_n_features] +
                                     this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}

void LinearRegressionModelForAdagrad::_update_w(vector<DatasetEntry>& batch){
    _copy(this->w_cur, this->w_prev);
    for(DatasetEntry& entry : batch){
        this->_update_w_one(entry);
    }
}
