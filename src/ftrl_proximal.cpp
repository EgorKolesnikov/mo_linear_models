#include "ftrl_proximal.h"
#include "reader.h"
#include "util.h"


LogisticRegressionModelForFtrlProximal::LogisticRegressionModelForFtrlProximal(
        int n_features = 0
        , int n_iterations = 3
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
    vector<vector<double> > z(this->_n_iterations, vector<double> (this->_n_features));
    vector<vector<double> > n(this->_n_iterations, vector<double> (this->_n_features));
    vector<vector<double> > w(this->_n_iterations, vector<double> (this->_n_features));
    for (int i = 0; i < this->_n_features; ++i) {
        n[0][i] = G[i] * G[i];
        z[0][i] = G[i] - n[0][i];
        w[0][i] = this->w_prev[i] * entry.x[i] * this->_learning_rate;
    }
    for (int i = 1; i < this->_n_iterations; ++i) {
        for (int j = 0; j < this->_n_features; ++j) {
            w[i][j] = -n[i - 1][j] * z[i - 1][j];
            w[i][j] = abs(w[i][j]);
            n[i][j] = n[i - 1][j] + G[j] * G[j];
            z[i][j] = z[i - 1][j] + G[j] - (1. / n[i][j] - 1 / n[i - 1][j]) * w[i][j];
        }
    }    
    double alpha = 1., grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->w_cur[i] = (this->w_prev[i] + w[this->_n_iterations - 1][i]) * alpha + this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps) - w[this->_n_iterations - 1][i];
    }
    G[this->_n_features] += grad * grad;
    this->w_cur[this->_n_features] = this->w_prev[this->_n_features] + this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}

void LogisticRegressionModelForFtrlProximal::_update_w_one_cache(DatasetEntry& entry, int idx_in_batch){
    vector<vector<double> > z(this->_n_iterations, vector<double> (this->_n_features));
    vector<vector<double> > n(this->_n_iterations, vector<double> (this->_n_features));
    vector<vector<double> > w(this->_n_iterations, vector<double> (this->_n_features));
    for (int i = 0; i < this->_n_features; ++i) {
        n[0][i] = G[i] * G[i];
        z[0][i] = G[i] - n[0][i];
        w[0][i] = this->w_prev[i] * entry.x[i] * this->_learning_rate;
    }
    for (int i = 1; i < this->_n_iterations; ++i) {
        for (int j = 0; j < this->_n_features; ++j) {
            w[i][j] = -n[i - 1][j] * z[i - 1][j];
            w[i][j] = abs(w[i][j]);
            n[i][j] = n[i - 1][j] + G[j] * G[j];
            z[i][j] = z[i - 1][j] + G[j] - (1. / n[i][j] - 1 / n[i - 1][j]) * w[i][j];
        }
    }    
    double alpha = 1., grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->parallel_batch_entries[idx_in_batch][i] = w[this->_n_iterations - 1][i] + (this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps) - w[this->_n_iterations - 1][i]) / alpha;
    }
    G[this->_n_features] += grad * grad;
    this->parallel_batch_entries[idx_in_batch][this->_n_features] = this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}

LinearRegressionModelForFtrlProximal::LinearRegressionModelForFtrlProximal(
        int n_features = 0
        , int n_iterations = 3
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
    vector<vector<double> > z(this->_n_iterations, vector<double> (this->_n_features));
    vector<vector<double> > n(this->_n_iterations, vector<double> (this->_n_features));
    vector<vector<double> > w(this->_n_iterations, vector<double> (this->_n_features));
    for (int i = 0; i < this->_n_features; ++i) {
        n[0][i] = G[i] * G[i];
        z[0][i] = G[i] - n[0][i];
        w[0][i] = this->w_prev[i] * entry.x[i] * this->_learning_rate;
    }
    for (int i = 1; i < this->_n_iterations; ++i) {
        for (int j = 0; j < this->_n_features; ++j) {
            w[i][j] = -n[i - 1][j] * z[i - 1][j];
            w[i][j] = abs(w[i][j]);
            n[i][j] = n[i - 1][j] + G[j] * G[j];
            z[i][j] = z[i - 1][j] + G[j] - (1. / n[i][j] - 1 / n[i - 1][j]) * w[i][j];
        }
    }    
    double alpha = 1., grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->w_cur[i] = (this->w_prev[i] + w[this->_n_iterations - 1][i]) * alpha + this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps) - w[this->_n_iterations - 1][i];
    }
    G[this->_n_features] += grad * grad;
    this->w_cur[this->_n_features] = this->w_prev[this->_n_features] + this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}

void LinearRegressionModelForFtrlProximal::_update_w_one_cache(DatasetEntry& entry, int idx_in_batch){
    vector<vector<double> > z(this->_n_iterations, vector<double> (this->_n_features));
    vector<vector<double> > n(this->_n_iterations, vector<double> (this->_n_features));
    vector<vector<double> > w(this->_n_iterations, vector<double> (this->_n_features));
    for (int i = 0; i < this->_n_features; ++i) {
        n[0][i] = G[i] * G[i];
        z[0][i] = G[i] - n[0][i];
        w[0][i] = this->w_prev[i] * entry.x[i] * this->_learning_rate;
    }
    for (int i = 1; i < this->_n_iterations; ++i) {
        for (int j = 0; j < this->_n_features; ++j) {
            w[i][j] = -n[i - 1][j] * z[i - 1][j];
            w[i][j] = abs(w[i][j]);
            n[i][j] = n[i - 1][j] + G[j] * G[j];
            z[i][j] = z[i - 1][j] + G[j] - (1. / n[i][j] - 1 / n[i - 1][j]) * w[i][j];
        }
    }    
    double alpha = 1., grad = this->_gradient_one(entry);
    for(int i = 0; i < this->_n_features; ++i){
        G[i] += grad * grad;
        this->parallel_batch_entries[idx_in_batch][i] = w[this->_n_iterations - 1][i] + (this->_learning_rate * grad * entry.x[i] / sqrt(G[i] + eps) - w[this->_n_iterations - 1][i]) / alpha;
    }
    G[this->_n_features] += grad * grad;
    this->parallel_batch_entries[idx_in_batch][this->_n_features] = this->_learning_rate * grad * 1.0 / sqrt(G[this->_n_features] + eps);
}