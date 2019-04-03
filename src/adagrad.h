#pragma once

#include "base.h"


class AdagradRegression : public TAbstractLinearModel {

public:
    AdagradRegression(
            int n_features
            , int _n_iterations
            , double _learning_rate
    );

    virtual void train(TFullDataReader&);
};
