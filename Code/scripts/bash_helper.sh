#!/bin/bash

for i in {0..2880}
do
    cd /home/ij405/ms_det/scripts
    mv *.json /scratch/ij405/thresh_models
    mv *.h5 /scratch/ij405/thresh_models
    sleep 5
done
