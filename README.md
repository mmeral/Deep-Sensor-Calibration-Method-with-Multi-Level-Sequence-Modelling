# Deep-Sensor-Calibration-Method-with-Multi-Level-Sequence-Modelling

This repository contains source code for semester project "Deep Sensor Calibration Method with Multi Level Sequence Modelling" in Computer Engineering and Networks Laboratory, TEC, Computer ETH Zürich.

*main_new.py* is the training script. For argument details, run 
     python main_new.py -h
     
*monthly_test_model_new.py* evaluates a saved model in monthly intervals with respect to mean absolute error. The argument must match with the arguments of the main_new.py during training. For details, run
     python main_new.py -h
 
*models_new.py* contains the model definitions.

Project Organization
------------

    ├── README.md          <- The top-level README
    ├── data               <- Directory for the data
    │   └── Beijing        
    │       └── processed     
    │           └── co-located
    │               └── locationName_compare.csv   <- Save the dataset for the location locationName as locationName_compare.csv 
    │                                                   The first column must be time, 5th column must be the raw sensor value of the aimed pollutant and last
    │                                                   column must be the ground truth pollutant concentration
    │ 
    ├── main_new.py                    <- Main training script to train NetSED and NetglobalSED variants
    │
    ├── models_new.py                  <- Model definitions of NetSED and NetglobalSED variants
    │
    ├── monthly_test_model_new.py      <- Display monthly mean absolute errors of calibrations of a saved model.
    │
    └── utils_new.py                   <- Utility functions
    


------------
