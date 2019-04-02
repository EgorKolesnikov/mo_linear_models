#include "args.h"


const string usage = 
	"\n"\
	"Usage: ./lm [train|test] [ALGO] [TASK] [FILE_PATH] [SAVE_MODEL_PATH]\n"\
	" - ALGO: 'sgd' | 'adagrad' | 'ftrl-proximal'\n"
	" - TASK: 'classification' | 'regression'\n";


int _parse_train_args(int argc, char * argv[], ArgWrap& args){
	if(argc == 2){
		printf("You need to specify algo type\n");
		return 2;
	}

	if(strcmp(argv[2], "sgd") == 0){
		args.algo_type = AlgoType::sgd;
	} else if(strcmp(argv[2], "adagrad") == 0) {
		args.algo_type = AlgoType::adagrad;
	} else if(strcmp(argv[2], "ftrl-proximal") == 0){
		args.algo_type = AlgoType::ftrl_proximal;
	} else {
		printf("Unknown algo type '%s'\n", argv[2]);
		return 2;
	}

	if(argc == 3){
		printf("You need to specify task type\n");
		return 2;
	}

	if(strcmp(argv[3], "classification") == 0){
		args.task_type = TaskType::classification;
	} else if(strcmp(argv[3], "regression") == 0) {
		args.task_type = TaskType::regression;
	} else {
		printf("Unknown task type '%s'\n", argv[3]);
		return 2;
	}

	if(argc == 4){
		printf("You need to specify input file path\n");
		return 2;
	}

	args.input_file_path = argv[4];

	if(argc == 5){
		printf("You need to specify model output path\n");
		return 2;
	}

	args.save_model_path = argv[5];
	return 0;
}

int _parse_test_args(int argc, char * argv[], ArgWrap& args){
	return 0;
}

int parse_args(int argc, char * argv[], ArgWrap& args){
	if(argc == 1){
		printf("Invalid arguments. Ckeckout usage.\n");
		return 1;
	}

	if(strcmp(argv[1], "train") == 0){
		return _parse_train_args(argc, argv, args);
	} else if(strcmp(argv[1], "test") == 0){
		return _parse_test_args(argc, argv, args);
	}

	printf("Expecting to have 'train' or 'test' as first argument. Found: '%s'\n", argv[1]);
	return 2;
}