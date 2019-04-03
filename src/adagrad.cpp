#include <cmath>

#include "adagrad.h"
#include "reader.h"


AdagradRegression::AdagradRegression(int n_features, int n_iterations, double learning_rate)
        : TAbstractLinearModel(n_features, n_iterations, learning_rate)
{ }

int sign(double arg);
vector<double> getPredictions(vector<DatasetEntry> data, vector<double> weights);
vector<double> test(vector<DatasetEntry> data, vector<double> weights);
vector<DatasetEntry> scale(vector<DatasetEntry> data);

void AdagradRegression::train(TFullDataReader& reader){
    printf("Starting adagrad train with %ld entries\n", reader.dataset.size());
    int T = 10;
    reader.dataset = scale(reader.dataset);
    int n_features = reader._n_features;
    vector<double> weights(n_features, 0);
    vector<double> G(n_features + 1, 0);
    vector<double> pred(reader._n_entries, 0);

    double free = 0;
    for (int t = 0; t < T; t++) {
        pred = getPredictions(reader.dataset, weights);
        for (int j = 0; j < n_features; j++) {
            double grad = 0;
            for (int i = 0; i < reader._n_entries; i++) {
                grad = reader.dataset[i].x[j] * (1 / (1 + exp(-pred[i] - free)) - reader.dataset[i].y);
                grad /= reader._n_entries;
                G[j] += grad * grad;
                weights[j] -= _learning_rate * grad / sqrt(G[j]);
            }
        }
        for (int i = 0; i < reader._n_entries; i++) {
            double grad = (1 / (1 + exp(-pred[i] - free)) - reader.dataset[i].y);
            grad /= reader._n_entries;
            G[n_features] += grad * grad;
            free -= _learning_rate * grad / sqrt(G[n_features]);
        }
    }
    //}
    weights.push_back(free);
    for (int i = 0; i < weights.size(); i++) {
        cout << weights[i] << std::endl;
    }

    double acc = 0;
    vector<double > y = test(reader.dataset, weights);
    for (int i = 0; i < y.size(); i++) {
        acc += y[i] == reader.dataset[i].y ? 1 : 0;
        cout << y[i] << " " << reader.dataset[i].y << std::endl;
    }
    acc /= y.size();
    cout << "accuracy " << acc << std::endl;

    printf("Done\n");
}

vector<double> getPredictions(vector<DatasetEntry> data, vector<double> weights) {
    int n_features = weights.size();
    vector<double> res(data.size());
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < n_features; j++) {
            res[i] += data[i].x[j] * weights[j];
        }
    }
    return res;
}

int sign(double arg) {
    if (arg > 0) {
        return 1;
    }
    return 0;
}

vector<double> test(vector<DatasetEntry> data, vector<double> weights) {
    vector<double> y;
    for (int i = 0; i < data.size(); i++) {
        y.push_back(0);
        for (int j = 0; j < data[i].x.size() - 1; j++) {
            y[i] += data[i].x[j] * weights[j];
        }
        y[i] += weights[data[0].x.size() - 1];
        y[i] = 1 / (1 + exp(-y[i]));
        y[i] = y[i] > 0.5 ? 1 : 0;
    }
    return y;
}

vector<DatasetEntry> scale(vector<DatasetEntry> data) {

    int n_features = data[0].x.size() - 1;
    for (int j = 0; j < n_features; j++) {
        double avg = 0;
        for (int i = 0; i < data.size(); i++) {
            avg += data[i].x[j];
        }
        avg /= data.size();
        double sigma = 0;
        for (int i = 0; i < data.size(); i++) {
            sigma += (data[i].x[j] - avg) * (data[i].x[j] - avg);
        }
        sigma = sqrt(sigma / data.size());
        for (int i = 0; i < data.size(); i++) {
            data[i].x[j] = (data[i].x[j] - avg) / sigma;
        }
    }
    return data;
}
