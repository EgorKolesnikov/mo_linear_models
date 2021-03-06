#include <iostream>
#include <string.h>

#include "reader.h"
#include "args.h"
#include "sgd.h"
#include "util.h"
#include "adagrad.h"
#include "ftrl_proximal.h"

using std::string;
using std::cout;


const int DEFAULT_N_ITERATIONS = 50000;
const double DEFAULT_LEARNING_RATE = 0.001;
const double DEFAULT_LEARNING_RATE_DECAY = 1.0;

const bool IS_PARALLEL = false;


int _run_sgd(const ArgWrap& args){
    // printf("Running SGD\n");

    TFullDataReader reader(args.data_path, true, true);
    if(!reader.is_open()){
        printf(" * SGD::FileLoad. File '%s' not open\n", args.data_path.c_str());
        return 1;
    }

    int n_iterations = DEFAULT_N_ITERATIONS;
    double learning_rate = DEFAULT_LEARNING_RATE;
    double learning_rate_decay = DEFAULT_LEARNING_RATE_DECAY;
    
    if(args.stage == Stage::train){
        if(args.task_type == TaskType::classification){
            printf(" * SGD::Classification::TRAIN. Fit LogisticRegressionModel\n");
            LogisticRegressionModel log_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay);

            if(IS_PARALLEL){
                log_reg.fit_parallel(reader);
            } else {
                log_reg.fit(reader, 1);
            }

            // printf(" * SGD::Classification::TRAIN. Saving model to '%s'\n", args.model_path.c_str());
            log_reg.save(args.model_path);
        } else {
            printf(" * SGD::Regression::TRAIN. Fit LinearRegressionModel\n");
            LinearRegressionModel lin_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay);
            
            if(IS_PARALLEL){
                lin_reg.fit_parallel(reader);
            } else {
                lin_reg.fit(reader, 1);
            }

            // printf(" * SGD::Regression::TRAIN. Saving model to '%s'\n", args.model_path.c_str());
            lin_reg.save(args.model_path);
        }
    } else {
        reader._is_circular = false;

        if(args.task_type == TaskType::classification){
            // printf(" * SGD::Classification::TEST. Loading LogisticRegressionModel\n");
            LogisticRegressionModel log_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay);
            log_reg.load(args.model_path);

            printf(" * SGD::Classification::TEST. Predicting and evaluating\n");
            double res = log_reg.evaluate(reader);
            
            printf(" * SGD::Classification::TEST. Result : %.5f\n", res);
        } else {
            // printf(" * SGD::Regression::TEST. Loading LinearRegressionModel\n");
            LinearRegressionModel lin_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay);
            lin_reg.load(args.model_path);

            printf(" * SGD::Regression::TEST. Predicting and evaluating\n");
            double res = lin_reg.evaluate(reader);

            printf(" * SGD::Regression::TEST. Result : %.5f\n", res);
        }
    }

    // printf("All done\n");
    return 0;
}

int _run_adagrad(const ArgWrap& args){
    // printf("Running adagrad\n");

    TFullDataReader reader(args.data_path, true, true);
    if(!reader.is_open()){
        printf(" * Adagrad::FileLoad. File '%s' not open\n", args.data_path.c_str());
        return 1;
    }

    int n_iterations = DEFAULT_N_ITERATIONS;
    double learning_rate = 0.5;
    double learning_rate_decay = 1.0 - 1e-8;

    double eps = 0.01;

    if(args.stage == Stage::train){
        if(args.task_type == TaskType::classification){
            printf(" * Adagrad::Classification::TRAIN. Fit LogisticRegressionModel\n");
            LogisticRegressionModelForAdagrad log_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay, eps);
            
            if(IS_PARALLEL){
                log_reg.fit_parallel(reader);
            } else {
                log_reg.fit(reader, 1);
            }

            // printf(" * Adagrad::Classification::TRAIN. Saving model to '%s'\n", args.model_path.c_str());
            log_reg.save(args.model_path);
        } else {
            printf(" * Adagrad::Regression::TRAIN. Fit LinearRegressionModel\n");
            LinearRegressionModelForAdagrad lin_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay, eps);
            
            if(IS_PARALLEL){
                lin_reg.fit_parallel(reader);
            } else {
                lin_reg.fit(reader, 1);
            }

            // printf(" * Adagrad::Regression::TRAIN. Saving model to '%s'\n", args.model_path.c_str());
            lin_reg.save(args.model_path);
        }
    } else {
        reader._is_circular = false;

        if(args.task_type == TaskType::classification){
            // printf(" * Adagrad::Classification::TEST. Loading LogisticRegressionModel\n");
            LogisticRegressionModelForAdagrad log_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay, eps);
            log_reg.load(args.model_path);

            printf(" * Adagrad::Classification::TEST. Predicting and evaluating\n");
            double res = log_reg.evaluate(reader);

            printf(" * Adagrad::Classification::TEST. Result : %.5f\n", res);
        } else {
            // printf(" * Adagrad::Regression::TEST. Loading LinearRegressionModel\n");
            LinearRegressionModelForAdagrad lin_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay, eps);
            lin_reg.load(args.model_path);

            printf(" * Adagrad::Regression::TEST. Predicting and evaluating\n");
            double res = lin_reg.evaluate(reader);

            printf(" * Adagrad::Regression::TEST. Result : %.5f\n", res);
        }
    }

    // printf("All done\n");
    return 0;
}

int _run_ftrl_proximal(const ArgWrap& args){
    // printf("Running ftrl_proximal\n");

    TFullDataReader reader(args.data_path, true, true);
    if(!reader.is_open()){
        printf(" * FtrlProximal::FileLoad. File '%s' not open\n", args.data_path.c_str());
        return 1;
    }

    int n_iterations = DEFAULT_N_ITERATIONS * 2;
    double learning_rate = 0.5;
    double learning_rate_decay = 1.0 - 1e-8;

    double eps = 0.01;

    if(args.stage == Stage::train){
        if(args.task_type == TaskType::classification){
            printf(" * FtrlProximal::Classification::TRAIN. Fit LogisticRegressionModel\n");
            LogisticRegressionModelForFtrlProximal log_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay, eps);
            
            if(IS_PARALLEL){
                log_reg.fit_parallel(reader);
            } else {
                log_reg.fit(reader, 1);
            }

            // printf(" * FtrlProximal::Classification::TRAIN. Saving model to '%s'\n", args.model_path.c_str());
            log_reg.save(args.model_path);
        } else {
            printf(" * FtrlProximal::Regression::TRAIN. Fit LinearRegressionModel\n");
            LinearRegressionModelForFtrlProximal lin_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay, eps);
            
            if(IS_PARALLEL){
                lin_reg.fit_parallel(reader);
            } else {
                lin_reg.fit(reader, 1);
            }

            // printf(" * FtrlProximal::Regression::TRAIN. Saving model to '%s'\n", args.model_path.c_str());
            lin_reg.save(args.model_path);
        }
    } else {
        reader._is_circular = false;

        if(args.task_type == TaskType::classification){
            // printf(" * FtrlProximal::Classification::TEST. Loading LogisticRegressionModel\n");
            LogisticRegressionModelForFtrlProximal log_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay, eps);
            log_reg.load(args.model_path);

            printf(" * FtrlProximal::Classification::TEST. Predicting and evaluating\n");
            double res = log_reg.evaluate(reader);

            printf(" * FtrlProximal::Classification::TEST. Result : %.5f\n", res);
        } else {
            // printf(" * FtrlProximal::Regression::TEST. Loading LinearRegressionModel\n");
            LinearRegressionModelForFtrlProximal lin_reg(reader._n_features, n_iterations, learning_rate, learning_rate_decay, eps);
            lin_reg.load(args.model_path);

            printf(" * FtrlProximal::Regression::TEST. Predicting and evaluating\n");
            double res = lin_reg.evaluate(reader);

            printf(" * FtrlProximal::Regression::TEST. Result : %.5f\n", res);
        }
    }

    // printf("All done\n");
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