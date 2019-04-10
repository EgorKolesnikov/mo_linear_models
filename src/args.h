#pragma once

#include <string.h>
#include <string>

using std::string;


/*
*	For now use dummy parser.
*	TODO: Boost.Options.
*/

enum class Stage {
	train,
	test
};

enum class AlgoType {
	sgd,
	adagrad,
	ftrl_proximal
};

enum class TaskType {
	classification,
	regression
};


struct ArgWrap{
	Stage stage;
	AlgoType algo_type;
	TaskType task_type;

	string data_path;
	string model_path;

	ArgWrap()
		: stage(Stage::train)
		, algo_type(AlgoType::sgd)
		, task_type(TaskType::classification)
		, data_path("")
		, model_path("")
	{ }
};

extern const string usage;

int parse_args(int argc, char * argv[], ArgWrap& args);
