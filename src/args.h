#pragma once

#include <string.h>
#include <string>

using std::string;


/*
*	For now use dummy parser.
*	TODO: Boost.Options.
*/

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
	AlgoType algo_type;
	TaskType task_type;

	char * input_file_path;
	char * save_model_path;

	ArgWrap()
		: algo_type(AlgoType::sgd)
		, task_type(TaskType::classification)
		, input_file_path(NULL)
		, save_model_path(NULL)
	{ }
};

extern const string usage;

int parse_args(int argc, char * argv[], ArgWrap& args);
