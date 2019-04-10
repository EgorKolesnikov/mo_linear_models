#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono> 

using std::cout;
using std::vector;
using std::string;
using std::ifstream;
using std::istringstream;
using std::shuffle;
using std::default_random_engine;


struct DatasetEntry {
	vector<double> x;
	double y;

	DatasetEntry()
		: x(0)
		, y(-1)
	{ }

	DatasetEntry(string& str, bool is_class_at_the_end);
};


class TAbstractDataReader{
	/*
	*	Basic interface of all readers.
	*	For now is not final, only for dummy one.
	*/
protected:
	ifstream inf;
	string _line;

public:
	// If true, then assuming that class label is at the end
	// of the file line. If false - at the beginning
	bool _is_class_at_the_end;

	// First line of input file should contain number of lines (dataset entries)
	// and number of features. Expecting simlpe csv file without header.
	int _n_entries;
	int _n_features;

	TAbstractDataReader(string path, bool is_class_at_the_end = true);
	~TAbstractDataReader();

	// To know, whether private self.inf has been opened successfully
	bool is_open();

	virtual vector<DatasetEntry> next_batch(size_t size = 32) = 0;
};


class TFullDataReader : public TAbstractDataReader {;
/*
*	Dummy file reader (just for checking).
*	Loading full file in memory.
*/
protected:
	int _iter;

	void _shuffle();
	void _restart();

public:
	vector<DatasetEntry> dataset;

	TFullDataReader(string path, bool is_class_at_the_end);

	vector<DatasetEntry> next_batch(size_t size = 32);
	vector<double> get_labels();

	// Load full dataset in self.dataset
	void load();
};
