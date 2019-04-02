#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using std::cout;
using std::vector;
using std::string;
using std::ifstream;
using std::istringstream;


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

	TAbstractDataReader(char * path, bool is_class_at_the_end);
	~TAbstractDataReader();

	// To know, whether private self.inf has been opened successfully
	bool is_open();
};


class TFullDataReader : public TAbstractDataReader {;
/*
*	Dummy file reader (just for checking).
*	Loading full file in memory.
*/
public:
	vector<DatasetEntry> dataset;

	TFullDataReader(char * path, bool is_class_at_the_end);

	// Load full dataset in self.dataset
	void load();
};
