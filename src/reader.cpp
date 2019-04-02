#include "reader.h"


/*
*	DatasetEntry utils
*/

DatasetEntry::DatasetEntry(string& str, bool is_class_at_the_end){
	/*
	*	Split given string by ',' anc convert each item to double.
	*	Line class label is at the end or at the beginning of the file
	*	(depending on is_class_at_the_end param).
	*	TODO: I'm sure it may be done better
	*/
	
	std::stringstream ss(str);

    double i;
    while (ss >> i){
        this->x.push_back(i);

        if (ss.peek() == ','){
            ss.ignore();
        }
    }

    this->y = this->x.back();
    this->x.pop_back();
}


/*
*	Abstract reader
*/

TAbstractDataReader::TAbstractDataReader(char * path, bool is_class_at_the_end = true)
	: inf(path)
	, _line("")
	, _is_class_at_the_end(is_class_at_the_end)
	, _n_entries(0)
	, _n_features(0)
{
	if(this->inf.is_open()){
		this->inf >> this->_n_entries;
		this->inf >> this->_n_features;
	}
}

TAbstractDataReader::~TAbstractDataReader(){
	this->inf.close();
}

bool TAbstractDataReader::is_open(){
	return this->inf.is_open();
}


/*
*	Full dataset reader
*/

TFullDataReader::TFullDataReader(char * path, bool is_class_at_the_end = true)
	: TAbstractDataReader(path, is_class_at_the_end)
{ 
	this->dataset.reserve(this->_n_entries);
}

void TFullDataReader::load(){
	while(this->inf >> this->_line){
		if(this->_line.empty()){
			break;
		}
		this->dataset.push_back(DatasetEntry(this->_line, this->_is_class_at_the_end));
	}
}
