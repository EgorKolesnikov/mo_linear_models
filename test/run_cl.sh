#!/bin/bash

echo "Train sgd with vowpal wabbit" 
time vw -d train_avazu_small.vw --loss_function logistic --sgd --binary --quiet -f model_vw_sgd.vw
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test sgd with vowpal wabbit" 
vw -i model_vw_sgd.vw -t --binary -d test_avazu_small.vw
sleep 5
echo " " 
echo " " 
echo " "

echo "Train adaptive with vowpal wabbit" 
time vw -d train_avazu_small.vw --loss_function logistic --adaptive --binary --quiet -f model_vw_sgd.vw
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test adaptive with vowpal wabbit" 
vw -i model_vw_sgd.vw -t --binary -d test_avazu_small.vw
sleep 5
echo " " 
echo " " 
echo " "

echo "Train ftrl with vowpal wabbit" 
time vw -d train_avazu_small.vw --loss_function logistic --ftrl --binary --quiet -f model_vw_sgd.vw
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test ftrl with vowpal wabbit" 
vw -i model_vw_sgd.vw -t --binary -d test_avazu_small.vw
sleep 5
echo " " 
echo " " 
echo " "

echo "Train with liblinear" 
time ~/liblinear-2.30/train -s 6 -q train_avazu_small.ll model_liblinear.ll
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test with liblinear" 
touch output.txt
~/liblinear-2.30/predict -b 1 test_avazu_small.ll model_liblinear.ll output.txt
sleep 5
echo "" 
echo " " 
echo " "

echo "Train sgd with our library"
time ~/mo_linear_models/bin/lm train sgd classification train_avazu_small.txt model.txt
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test sgd with our library"
~/mo_linear_models/bin/lm test sgd classification test_avazu_small.txt model.txt
sleep 5
echo " " 
echo " " 
echo " "

echo "Train adagrad with our library"
time ~/mo_linear_models/bin/lm train adagrad classification train_avazu_small.txt model.txt
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test adagrad with our library"
~/mo_linear_models/bin/lm test adagrad classification test_avazu_small.txt model.txt
sleep 5
echo " " 
echo " " 
echo " "

echo "Train ftrl-poximal with our library"
time ~/mo_linear_models/bin/lm train ftrl-proximal classification train_avazu_small.txt model.txt
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test ftrl-poximal with our library"
~/mo_linear_models/bin/lm test ftrl-proximal classification test_avazu_small.txt model.txt
