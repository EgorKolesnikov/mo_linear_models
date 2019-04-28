USE_DATASET="classification.3d.small.not_linear_separable.txt"
TASK="classification"

echo
echo "--- Train sgd"
../bin/lm train sgd "${TASK}" "./datasets/${USE_DATASET}" "./model.out"
echo "--- Test sgd"
../bin/lm test sgd "${TASK}" "./datasets/${USE_DATASET}" "./model.out"

# echo
# echo "--- Train adagrad"
# ../bin/lm train adagrad "${TASK}" "./datasets/${USE_DATASET}" "./model.out"
# echo "--- Test adagrad"
# ../bin/lm test adagrad "${TASK}" "./datasets/${USE_DATASET}" "./model.out"

# echo
# echo "--- Train ftrl-proximal"
# ../bin/lm train ftrl-proximal "${TASK}" "./datasets/${USE_DATASET}" "./model.out"
# echo "--- Test ftrl-proximal"
# ../bin/lm test ftrl-proximal "${TASK}" "./datasets/${USE_DATASET}" "./model.out"
