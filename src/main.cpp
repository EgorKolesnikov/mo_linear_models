#include "reader.h"
#include <iostream>
#include <string.h>


enum class AlgoType {
	sgd,
	adagrad,
	ftrl_proximal
};


struct ArgWrap{
	AlgoType type;

	ArgWrap()
		: type(AlgoType::sgd)
	{ }
};


int parse_args(int argc, char * argv[], ArgWrap& args){
	if(argc == 1){
		printf("Expected to have at least one argument: ['sgd', 'adagrad', 'ftrl-proximal']\n");
		return 1;
	}

	if(strcmp(argv[1], "sgd") == 0){
		args.type = AlgoType::sgd;
	} else if(strcmp(argv[1], "adagrad") == 0) {
		args.type = AlgoType::adagrad;
	} else if(strcmp(argv[1], "ftrl-proximal") == 0){
		args.type = AlgoType::ftrl_proximal;
	} else {
		printf("Unknown type '%s'\n", argv[1]);
		return 2;
	}

	return 0;
}

int run(const ArgWrap& args){
	if(args.type == AlgoType::sgd){
		printf("Running sgd\n");
	} else if(args.type == AlgoType::adagrad){
		printf("Running adagrad\n");
	} else if(args.type == AlgoType::ftrl_proximal){
		printf("Running ftrl_proximal\n");
	}
	return 0;
}


int main(int argc, char * argv[]){
	ArgWrap args;
	
	int rc_args = parse_args(argc, argv, args);
	if(rc_args){
		exit(rc_args);
	}

	int rc_run = run(args);
	if(rc_run){
		exit(rc_run);
	}

	return 0;
}
