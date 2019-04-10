#include <iostream>
#include <string.h>

#include "reader.h"
#include "args.h"
#include "sgd.h"
#include "util.h"

using std::string;
using std::cout;


int _run_sgd(const ArgWrap& args){
	printf("Running sgd\n");

	TFullDataReader reader(args.data_path, true);
	if(!reader.is_open()){
		printf("File '%s' not open\n", args.data_path.c_str());
		return 1;
	}

	reader.load();
	printf("Loaded %d dataset objects\n", int(reader.dataset.size()));
	
	if(args.stage == Stage::train){
		printf(" * Trainig SGD\n");
		SgdClassification sgd(reader._n_features, 5, 0.01, 1.0);
		sgd.fit(reader);

		printf(" * Saving model to '%s'\n", args.model_path.c_str());
		sgd.save(args.model_path);
	} else {
		printf(" * Loading model\n");
		SgdClassification sgd(reader._n_features, 5, 0.01, 1.0);
		sgd.load(args.model_path);

		printf(" * Predicting and evaluating\n");
		double res = sgd.evaluate(reader.dataset);
		printf(" * Result : %.2f\n", res);
	}

	printf("All done\n");
	return 0;
}

int _run_adagrad(const ArgWrap& args){
	printf("Running adagrad\n");
	return 0;
}

int _run_ftrl_proximal(const ArgWrap& args){
	printf("Running ftrl_proximal\n");
	return 0;
}

int run(const ArgWrap& args){
	if(args.algo_type == AlgoType::sgd){
		return _run_sgd(args);
	} else if(args.algo_type == AlgoType::adagrad){
		return _run_adagrad(args);
	} else if(args.algo_type == AlgoType::ftrl_proximal){
		return _run_ftrl_proximal(args);
	}

	printf("Unknown algo type. Terminating.\n");
	return 1;
}


int main(int argc, char * argv[]){
	ArgWrap args;
	
	int rc_args = parse_args(argc, argv, args);
	if(rc_args){
		cout << usage << "\n";
		exit(rc_args);
	}

	int rc_run = run(args);
	if(rc_run){
		cout << usage << "\n";
		exit(rc_run);
	}

	return 0;
}
