#pragma once

#include "base.h"


class SgdClassification : public TAbstractLinearModel {

protected:
	void _update_rule(vector<DatasetEntry>& batch);

public:
	SgdClassification(
		int n_features
		, int _n_iterations
		, double _learning_rate
		, double learning_rate_decay
	);
};
