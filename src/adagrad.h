#pragma once

#include <math.h>
#include <cmath>

#include "base.h"
#include "sgd.h"


class LogisticRegressionModelForAdagrad : public LogisticRegressionModel {

protected:
    virtual void _update_w_one_direct(DatasetEntry& entry);
    virtual void _update_w_one_cache(DatasetEntry& entry, int idx_in_batch);

public:
    vector<double> temp;
    vector<double> G;
    double eps;

    LogisticRegressionModelForAdagrad(
            int n_features
            , int _n_iterations
            , double _learning_rate
            , double learning_rate_decay
            , double eps
    );
};

class LinearRegressionModelForAdagrad : public LinearRegressionModel {

protected:
    virtual void _update_w_one_direct(DatasetEntry& entry);
    virtual void _update_w_one_cache(DatasetEntry& entry, int idx_in_batch);

public:
    vector<double> temp;
    vector<double> G;
    double eps;

    LinearRegressionModelForAdagrad(
            int n_features
            , int _n_iterations
            , double _learning_rate
            , double learning_rate_decay
            , double eps
    );
};