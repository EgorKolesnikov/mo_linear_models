RUN_SGD=true
RUN_ADAGRAD=true
RUN_FTRL_PROXIMAL=true


function _run_classification(){
	if [ "${RUN_SGD}" == true ]; then
		echo
		echo "--- Train sgd '$1'"
		../bin/lm train sgd "classification" "./datasets/$1" "./model.out"
		echo "--- Test sgd"
		../bin/lm test sgd "classification" "./datasets/$1" "./model.out"
	fi

	if [ "${RUN_ADAGRAD}" == true ]; then
		echo
		echo "--- Train adagrad"
		../bin/lm train adagrad "classification" "./datasets/$1" "./model.out"
		echo "--- Test adagrad"
		../bin/lm test adagrad "classification" "./datasets/$1" "./model.out"
	fi

	if [ "${RUN_FTRL_PROXIMAL}" == true ]; then
		echo
		echo "--- Train ftrl-proximal"
		../bin/lm train ftrl-proximal "classification" "./datasets/$1" "./model.out"
		echo "--- Test ftrl-proximal"
		../bin/lm test ftrl-proximal "classification" "./datasets/$1" "./model.out"
	fi
}


function _run_regression(){
	if [ "${RUN_SGD}" == true ]; then
		echo
		echo "--- Train sgd '$1'"
		../bin/lm train sgd "regression" "./datasets/$1" "./model.out"
		echo "--- Test sgd"
		../bin/lm test sgd "regression" "./datasets/$1" "./model.out"
	fi

	if [ "${RUN_ADAGRAD}" == true ]; then
		echo
		echo "--- Train adagrad"
		../bin/lm train adagrad "regression" "./datasets/$1" "./model.out"
		echo "--- Test adagrad"
		../bin/lm test adagrad "regression" "./datasets/$1" "./model.out"
	fi

	if [ "${RUN_FTRL_PROXIMAL}" == true ]; then
		echo
		echo "--- Train ftrl-proximal"
		../bin/lm train ftrl-proximal "regression" "./datasets/$1" "./model.out"
		echo "--- Test ftrl-proximal"
		../bin/lm test ftrl-proximal "regression" "./datasets/$1" "./model.out"
	fi
}


echo
echo '============== CLASS. Not linear separable ===================='
_run_classification "classification.3d.small.not_linear_separable.txt"

echo
echo '============== CLASS. Linear separable ===================='
_run_classification "classification.3d.small.linear_separable.txt"

echo
echo '============== REG. Straight line ===================='
_run_regression "regression.1d.small.straight_line.txt"

echo
echo '============== REG. Not straight line ===================='
_run_regression "regression.1d.small.not_straight_line.txt"
echo
