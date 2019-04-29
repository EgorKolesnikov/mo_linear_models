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

void LogisticRegressionModelForAdagrad::_update_w_one_direct(DatasetEntry& entry){
    double grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        double wi_grad = grad * entry.x[i];
        G[i] += wi_grad * wi_grad;
        this->w_cur[i] = this->w_prev[i] + this->_learning_rate * wi_grad / sqrt(this->G[i] + this->eps);
    }
    G[this->_n_features] += (grad * 1.0) * (grad * 1.0);
    this->w_cur[this->_n_features] = this->w_prev[this->_n_features] + this->_learning_rate * (grad * 1.0) / sqrt(this->G[this->_n_features] + this->eps);
}

void LogisticRegressionModelForAdagrad::_update_w_one_cache(DatasetEntry& entry, int idx_in_batch){
    double grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        double wi_grad = grad * entry.x[i];
        G[i] += wi_grad * wi_grad;
        this->parallel_batch_entries[idx_in_batch][i] = this->_learning_rate * wi_grad / sqrt(this->G[i] + this->eps);
    }
    G[this->_n_features] += (grad * 1.0) * (grad * 1.0);
    this->parallel_batch_entries[idx_in_batch][this->_n_features] = this->_learning_rate * (grad * 1.0) / sqrt(this->G[this->_n_features] + this->eps);
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

void LinearRegressionModelForAdagrad::_update_w_one_direct(DatasetEntry& entry){
    double grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        double wi_grad = grad * entry.x[i];
        G[i] += wi_grad * wi_grad;
        this->w_cur[i] = this->w_prev[i] + this->_learning_rate * wi_grad / sqrt(this->G[i] + this->eps);
    }
    G[this->_n_features] += (grad * 1.0) * (grad * 1.0);
    this->w_cur[this->_n_features] = this->w_prev[this->_n_features] + this->_learning_rate * (grad * 1.0) / sqrt(G[this->_n_features] + eps);
}

void LinearRegressionModelForAdagrad::_update_w_one_cache(DatasetEntry& entry, int idx_in_batch){
    double grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        double wi_grad = grad * entry.x[i];
        G[i] += wi_grad * wi_grad;
        this->parallel_batch_entries[idx_in_batch][i] = this->_learning_rate * wi_grad / sqrt(G[i] + eps);
    }
    G[this->_n_features] += (grad * 1.0) * (grad * 1.0);
    this->parallel_batch_entries[idx_in_batch][this->_n_features] = this->_learning_rate * (grad * 1.0) / sqrt(G[this->_n_features] + eps);
}