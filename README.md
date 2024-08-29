# Parametrized generation of directional antennas using a conditional Generative Adversarial Network
## Forschungsprojekt 2024

Source code for the paper "Parametrized generation of directional antennas using a conditional Generative Adversarial Network" 

### Requirements
- Python 3.11
- numpy 1.26.4
- matplotlib 3.9.1.post1
- necpp 1.7.3.5
- keras 3.5.0
- tensorflow 2.17.0
- tqdm 4.30.0

### data_creator
Code to generate training data. The folders ```indata```, ```generatedData```, ```changedExcitation``` and ```labeledData``` must be created in this directory to use the scripts. To generate training data, first generate the base antennas with the scripts ```mkDipole```, ```mkMoxon```, ```mkYagi``` and ```mkQuad```. The results will be generated in ```indata```. Then, run the scripts ```dataGenerator``` and ```dataGeneratorExcitation``` to generate all the permutations of the base antennas into ```generatedData``` / ```changedExcitation```. ```labelAll``` and ```labelAllExcitation``` will simulate the antennas and label them with the results, putting them in ```labeledData```. These scripts require a lot of handholding since the nec++ simulator often crashes / hangs. Finally the ```normalizer``` script will normalize the results and export them to ```normalized_output.csv```.

### validation_data_creator
Similar to the data_creator, but for the validation data. Requires the folders ```validation_data``` and ```labeled_validation_data```. First use ```mkBiQuadValidation```, ```mkDipoleValidation``` and ```mkMoxonValidation``` to generate the validation antennas. ```labelAllValidation``` will simulate the antennas and label them with the results, putting them in ```labeled_validation_data```. Finally ```normalizerValidation``` will normalize the results and export them to ```normalized_validation_output.csv```.

### train_model
Contains scripts to train various models with the data. The script ```train_cgan3d_keras.py``` was used to train the model presented in the paper, located in ```model```.```train_cgan3d_excitation_embedded.py``` contains a model that tries to use the excitation with embedding, but due to time constraints training didnt happen. The other scripts are older and were used to train previous iterations of models which were not successful / where the generator was unable to "beat" the discriminator.

### use_model
These scripts are used to manually validate the results from the trained model. ```use_generator``` will generate 10 antennas from labels out of ```normalized_output.csv``` and simulate them afterwards, outputting the labels and results. It will also show the generated geometry. ```use_generator_manually``` generates antennas from the labels in the script and shows the output. ```use_validator``` will use the discriminator on the generated validation data and output the results. The model from the paper is located in ```model```.