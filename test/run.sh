function classification(){
	USE_DATASET="classification.2d.small.linear_separable.txt"

	echo
	echo "--- Train"
	echo
	../bin/lm train sgd classification "./datasets/${USE_DATASET}" "./model.out"

	echo
	echo "--- Test"
	echo
	../bin/lm test sgd classification "./datasets/${USE_DATASET}" "./model.out"
}


function regression(){
	USE_DATASET="regression.1d.small.not_straight_line.txt"

	echo
	echo "--- Train"
	echo
	../bin/lm train sgd regression "./datasets/${USE_DATASET}" "./model.out"

	echo
	echo "--- Test"
	echo
	../bin/lm test sgd regression "./datasets/${USE_DATASET}" "./model.out"
}


classification
