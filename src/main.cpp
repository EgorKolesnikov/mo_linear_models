#include <iostream>
#include <string.h>

#include "reader.h"
#include "args.h"
#include "sgd.h"

using std::string;
using std::cout;


int _run_sgd(const ArgWrap& args){
	printf("Running sgd\n");

	TFullDataReader reader(args.input_file_path, true);
	if(!reader.is_open()){
		printf("File '%s' not open\n", args.input_file_path);
		return 1;
	}

	reader.load();
	cout << reader.dataset.size() << "\n";
	
	SgdClassification sgd(reader._n_features, 5, 0.01);
	sgd.train(reader);

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
