#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "reader.h"

using std::cout;
using std::vector;
using std::string;
using std::ifstream;
using std::istringstream;

/*
*	Basic vector-wise operations
*/

void _add(vector<double>& a, const vector<double>& b);
void _sub(vector<double>& a, const vector<double>& b);
void _mul(vector<double>& a, int value);
void _div(vector<double>& a, int value);
void _copy(const vector<double>& src, vector<double>& dst);


vector<double> grad(vector<double>& w, DatasetEntry& x, double reg = 0.0);

vector<double> batch_grad(vector<double>& w, vector<DatasetEntry>& batch, double reg = 0.0);
