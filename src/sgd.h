#pragma once

#include "base.h"


class SgdClassification : public TAbstractLinearModel {

public:
	SgdClassification(
		int n_features
		, int _n_iterations
		, double _learning_rate
	);

	virtual void train(TFullDataReader&);
};
