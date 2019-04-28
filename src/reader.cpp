#include "reader.h"


/*
*	DatasetEntry utils
*/

DatasetEntry::DatasetEntry(string& str, bool is_class_at_the_end){
	/*
	*	Split given string by ',' anc convert each item to double.
	*	Line class label is at the end or at the beginning of the file
	*	(depending on is_class_at_the_end param).
	*	TODO: I'm sure it may be done in more efficient way.
	*/
	
	std::stringstream ss(str);

    double i = 0.0;
    int count = 0;

    while (ss >> i){
    	if(count == 0 && !is_class_at_the_end){
    		this->y = i;
    	} else {
    		this->x.push_back(i);
    	}

        if (ss.peek() == ','){
            ss.ignore();
        }
    }

    if(is_class_at_the_end){
	    this->y = this->x.back();
	    this->x.pop_back();
	}
}


/*
*	Abstract reader
*/

TAbstractDataReader::TAbstractDataReader(string _path, bool is_class_at_the_end, bool is_circular)
	: path(_path)
	, inf(_path)
	, _line("")
	, _is_class_at_the_end(is_class_at_the_end)
	, _is_circular(is_circular)
	, _n_entries(0)
	, _n_features(0)
{
	this->to_the_begining();
}

TAbstractDataReader::~TAbstractDataReader(){
	this->inf.close();
}

void TAbstractDataReader::to_the_begining(){
	if(this->inf.eof()){
		this->inf.clear();
		this->inf.seekg(0, ios::beg);
	}

	this->inf >> this->_n_entries;
	this->inf >> this->_n_features;
}

bool TAbstractDataReader::is_open(){
	return this->inf.is_open();
}


/*
*	Full dataset reader
*/

TFullDataReader::TFullDataReader(string path, bool is_class_at_the_end, bool is_circular)
	: TAbstractDataReader(path, is_class_at_the_end, is_circular)
	, _iter(0)
{ 
	this->dataset.reserve(this->_n_entries);
}

void TFullDataReader::_shuffle(){
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  	shuffle(this->dataset.begin(), this->dataset.end(), std::default_random_engine(seed));
}

void TFullDataReader::_restart(){
	this->_shuffle();
	this->_iter = 0;
}

void TFullDataReader::load(){
	while(this->inf >> this->_line){
		if(this->_line.empty()){
			break;
		}
		this->dataset.push_back(DatasetEntry(this->_line, this->_is_class_at_the_end));
	}

	this->_restart();
}

vector<DatasetEntry> TFullDataReader::next_batch(size_t size){
	vector<DatasetEntry> batch;
	batch.reserve(size);	
	
	while(batch.size() < size && this->inf >> this->_line){
        if(this->_line.empty()){
            break;
        }
        batch.push_back(DatasetEntry(this->_line, this->_is_class_at_the_end));
		this->_iter %= this->_n_entries;
	}
    
    if(this->inf.eof() && this->_is_circular){
    	this->to_the_begining();
    }

	return batch;
}


vector<double> TFullDataReader::get_labels(){
	vector<double> labels;
	labels.reserve(this->dataset.size());

	for(DatasetEntry& e : this->dataset){
		labels.push_back(e.y);
	}

	return labels;
}