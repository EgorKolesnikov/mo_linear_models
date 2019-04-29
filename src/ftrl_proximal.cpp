#include "ftrl_proximal.h"
#include "reader.h"
#include "util.h"


LogisticRegressionModelForFtrlProximal::LogisticRegressionModelForFtrlProximal(
        int n_features = 0
        , int n_iterations = 10 
        , double learning_rate = 0.01
        , double learning_rate_decay = 2.0
        , double eps = 0.01
)
        : LogisticRegressionModel(n_features, n_iterations, learning_rate, learning_rate_decay)
        , temp(_n_features + 1, 0.0)
        , G(_n_features + 1, 0.0)
        , eps(eps)
{ }

void LogisticRegressionModelForFtrlProximal::_update_w_one_direct(DatasetEntry& entry){
    int size = this->_n_features;
    int iter = this->_n_iterations;
    vector<double> z_prev(size), z_cur(size);
    vector<double> n_prev(size), n_cur(size);
    vector<double> w_prev(size), w_cur(size);
    for (int i = 0; i < size; ++i) n_prev[i] = G[i] * G[i], z_prev[i] = G[i] - n_prev[i], w_prev[i] = this->w_prev[i] * entry.x[i] * this->_learning_rate;
    for (int i = 1; i < iter; ++i) {
        for (int j = 0; j < size; ++j) {
            w_cur[j] = abs(-n_prev[j] * z_prev[j]);
            n_cur[j] = n_prev[j] + G[j] * G[j]; 
            z_cur[j] = z_prev[j] + G[j] - (1. / n_cur[j] - 1. / n_prev[j]) * w_cur[j];
        }    
        for (int j = 0; j < size; ++j) z_prev[j] = z_cur[j], n_prev[j] = n_cur[j], w_prev[j] = w_cur[j]; break;
    }
    double alpha = 1., grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->w_cur[i] = (this->w_prev[i] + w_cur[i]) * alpha + this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps) - w_prev[i];
    }
    G[this->_n_features] += grad * grad;
    this->w_cur[this->_n_features] = this->w_prev[this->_n_features] + this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}

void LogisticRegressionModelForFtrlProximal::_update_w_one_cache(DatasetEntry& entry, int idx_in_batch){
    int size = this->_n_features;
    int iter = this->_n_iterations;
    vector<double> z_prev(size), z_cur(size);
    vector<double> n_prev(size), n_cur(size);
    vector<double> w_prev(size), w_cur(size);
    for (int i = 0; i < size; ++i) n_prev[i] = G[i] * G[i], z_prev[i] = G[i] - n_prev[i], w_prev[i] = this->w_prev[i] * entry.x[i] * this->_learning_rate;
    for (int i = 1; i < iter; ++i) {
        for (int j = 0; j < size; ++j) {
            w_cur[j] = abs(-n_prev[j] * z_prev[j]);
            n_cur[j] = n_prev[j] + G[j] * G[j]; 
            z_cur[j] = z_prev[j] + G[j] - (1. / n_cur[j] - 1. / n_prev[j]) * w_cur[j];
        }    
        for (int j = 0; j < size; ++j) z_prev[j] = z_cur[j], n_prev[j] = n_cur[j], w_prev[j] = w_cur[j]; break;
    }
    double alpha = 1., grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->parallel_batch_entries[idx_in_batch][i] = w_prev[i] + (this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps) - w_cur[i]) / alpha;
    }
    G[this->_n_features] += grad * grad;
    this->parallel_batch_entries[idx_in_batch][this->_n_features] = this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}

LinearRegressionModelForFtrlProximal::LinearRegressionModelForFtrlProximal(
        int n_features = 0
        , int n_iterations = 10
        , double learning_rate = 0.01
        , double learning_rate_decay = 1.0
        , double eps = 0.01
)
        : LinearRegressionModel(n_features, n_iterations, learning_rate, learning_rate_decay)
        , temp(_n_features + 1, 0.0)
        , G(_n_features + 1, 0.0)
        , eps(eps)
{ }

void LinearRegressionModelForFtrlProximal::_update_w_one_direct(DatasetEntry& entry){
    int size = this->_n_features;
    int iter = this->_n_iterations;
    vector<double> z_prev(size), z_cur(size);
    vector<double> n_prev(size), n_cur(size);
    vector<double> w_prev(size), w_cur(size);
    for (int i = 0; i < size; ++i) n_prev[i] = G[i] * G[i], z_prev[i] = G[i] - n_prev[i], w_prev[i] = this->w_prev[i] * entry.x[i] * this->_learning_rate;
    for (int i = 1; i <iter; ++i) {
        for (int j = 0; j < size; ++j) {
            w_cur[j] = abs(-n_prev[j] * z_prev[j]);
            n_cur[j] = n_prev[j] + G[j] * G[j]; 
            z_cur[j] = z_prev[j] + G[j] - (1. / n_cur[j] - 1. / n_prev[j]) * w_cur[j];
        }    
        for (int j = 0; j < size; ++j) z_prev[j] = z_cur[j], n_prev[j] = n_cur[j], w_prev[j] = w_cur[j]; break;
    }
    double alpha = 1., grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->w_cur[i] = (this->w_prev[i] + w_prev[i]) * alpha + this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps) - w_cur[i];
    }
    G[this->_n_features] += grad * grad;
    this->w_cur[this->_n_features] = this->w_prev[this->_n_features] + this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}

void LinearRegressionModelForFtrlProximal::_update_w_one_cache(DatasetEntry& entry, int idx_in_batch){
    int size = this->_n_features;
    int iter = this->_n_iterations;
    vector<double> z_prev(size), z_cur(size);
    vector<double> n_prev(size), n_cur(size);
    vector<double> w_prev(size), w_cur(size);
    for (int i = 0; i < size; ++i) n_prev[i] = G[i] * G[i], z_prev[i] = G[i] - n_prev[i], w_prev[i] = this->w_prev[i] * entry.x[i] * this->_learning_rate;
    for (int i = 1; i < iter; ++i) {
        for (int j = 0; j < size; ++j) {
            w_cur[j] = abs(-n_prev[j] * z_prev[j]);
            n_cur[j] = n_prev[j] + G[j] * G[j]; 
            z_cur[j] = z_prev[j] + G[j] - (1. / n_cur[j] - 1. / n_prev[j]) * w_cur[j];
        }    
        for (int j = 0; j < size; ++j) z_prev[j] = z_cur[j], n_prev[j] = n_cur[j], w_prev[j] = w_cur[j]; break;
    }
    double alpha = 1., grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->parallel_batch_entries[idx_in_batch][i] = w_prev[i] + (this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps) - w_cur[i]) / alpha;
    }
    G[this->_n_features] += grad * grad;
    this->parallel_batch_entries[idx_in_batch][this->_n_features] = this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}
