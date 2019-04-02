#include "sgd.h"
#include "reader.h"


SgdClassification::SgdClassification(int n_features = 0, int n_iterations = 5, double learning_rate = 0.01)
	: TAbstractLinearModel(n_features, n_iterations, learning_rate)
{ }


void SgdClassification::train(TFullDataReader& reader){
	printf("Starting sgd train with %ld entries\n", reader.dataset.size());
	printf("Done\n");
}
