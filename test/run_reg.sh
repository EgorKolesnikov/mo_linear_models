#!/bin/bash

echo "Train sgd with vowpal wabbit" 
time vw -d train_reg.vw --loss_function squared --sgd --quiet -f model_vw_sgd.vw
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "Test sgd with vowpal wabbit" 
vw -i model_vw_sgd.vw -t -d test_reg.vw
sleep 5
echo " " 
echo " " 
echo " " 

echo "Train adaptive with vowpal wabbit" 
time vw -d train_reg.vw --loss_function squared --adaptive --quiet -f model_vw_sgd.vw
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "Test adaptive with vowpal wabbit" 
vw -i model_vw_sgd.vw -t -d test_reg.vw
sleep 5
echo " " 
echo " " 
echo " " 

echo "Train ftrl with vowpal wabbit" 
time vw -d train_reg.vw --loss_function squared --ftrl --quiet -f model_vw_sgd.vw
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "Test ftrl with vowpal wabbit"
vw -i model_vw_sgd.vw -t -d test_reg.vw
sleep 5
echo " " 
echo " " 
echo " " 

echo "Train with liblinear" 
time ~/liblinear-2.30/train -s 11 -p 0 -q train_reg.ll model_liblinear.ll
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "Test with liblinear" 
touch output.txt
~/liblinear-2.30/predict test_reg.ll model_liblinear.ll output.txt
sleep 5
echo " " 
echo " " 
echo " " 

echo "Train sgd with our library"
time ~/mo_linear_models/bin/lm train sgd regression train_reg.txt model.txt
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test sgd with our library"
~/mo_linear_models/bin/lm test sgd regression test_reg.txt model.txt
sleep 5
echo " " 
echo " " 
echo " "

echo "Train adagrad with our library"
time ~/mo_linear_models/bin/lm train adagrad regression train_reg.txt model.txt
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test adagrad with our library"
~/mo_linear_models/bin/lm test adagrad regression test_reg.txt model.txt
sleep 5
echo " " 
echo " " 
echo " "

echo "Train ftrl-poximal with our library"
time ~/mo_linear_models/bin/lm train ftrl-proximal regression train_reg.txt model.txt
sleep 5
echo "---------------------------------------------" 
echo "---------------------------------------------" 
echo "---------------------------------------------"
echo "Test ftrl-poximal with our library"
~/mo_linear_models/bin/lm test ftrl-proximal regression test_reg.txt model.txt
