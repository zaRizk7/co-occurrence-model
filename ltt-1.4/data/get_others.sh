#!/bin/sh

# 100 words of 20 newsgroups
curl -o 20news_w100.mat http://www.cs.toronto.edu/~roweis/data/20news_w100.mat

# UCI CAR EVALUATION
curl -o car.data http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data
python convert.py car.data

# COIL-86
curl -o ticdata2000.txt http://kdd.ics.uci.edu/databases/tic/ticdata2000.txt
curl -o ticeval2000.txt http://kdd.ics.uci.edu/databases/tic/ticeval2000.txt
curl -o tictgts2000.txt http://kdd.ics.uci.edu/databases/tic/tictgts2000.txt

# COIL-46 (from Nevin L. Zhang)

curl -o coilDataTrain.txt http://www.cs.ust.hk/faculty/lzhang/hlcmResources/coilData/coilDataTrain.txt
curl -o coilDataTest.txt http://www.cs.ust.hk/faculty/lzhang/hlcmResources/coilData/coilDataTest.txt
grep "^[0123456789]" coilDataTrain.txt > coilDataTrain_matlab.txt
grep "^[0123456789]" coilDataTest.txt > coilDataTest_matlab.txt
