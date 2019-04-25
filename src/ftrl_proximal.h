#pragma once

#include <math.h>
#include <cmath>

#include "base.h"
#include "sgd.h"


class LogisticRegressionModelForFtrlProximal : public LogisticRegressionModel {

protected:
    virtual void _update_w_one(DatasetEntry& entry);
    virtual void _update_w(vector<DatasetEntry>& batch);

public:
    vector<double> temp;
    vector<double> G;
    double eps;

    LogisticRegressionModelForFtrlProximal(
            int n_features
            , int _n_iterations
            , double _learning_rate
            , double learning_rate_decay
            , double eps
    );
};

class LinearRegressionModelForFtrlProximal : public LinearRegressionModel {

protected:
    virtual void _update_w_one(DatasetEntry& entry);
    virtual void _update_w(vector<DatasetEntry>& batch);

public:
    vector<double> temp;
    vector<double> G;
    double eps;

    LinearRegressionModelForFtrlProximal(
            int n_features
            , int _n_iterations
            , double _learning_rate
            , double learning_rate_decay
            , double eps
    );
};
