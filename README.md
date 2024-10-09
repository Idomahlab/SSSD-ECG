# Deep Learning Project for Arrhythmia Classification  enhancment using Diffusing model

Overview
This project involves training a deep learning classifier to identify arrhythmias from ECG data in the PTB-XL database. The project also includes generating synthetic samples using a diffusion model to enhance the performance of the classifier. The steps below guide you through downloading the dataset, combining data, training the classifier, generating synthetic samples, and evaluating the classifier's performance.
Setup
1. Download the PTB-XL Database

Download the PTB-XL data from the following link: [PTB-XL Database](https://figshare.com/s/43df16e4a50e4dd0a0c5).

2. Combine Train and Validation Sets

Combine the training and validation sets using the combine_data.py script:

bash
Copy code
python combine_data.py --input_dir path_to_ptbxl_data --output_dir path_to_combined_data
3. Train the Basic Classifier

Train the basic classifier using the train_res50.py script:

bash
Copy code
python train_res50.py --data_dir path_to_combined_data --output_dir path_to_model
4. Evaluate the Classifier on the Test Set

Run the trained classifier on the test set and get the performance report using the test_res50.py script:

bash
Copy code
python test_res50.py --model_dir path_to_model --test_data_dir path_to_test_data
5. Choose a Label to Enhance

Choose an arrhythmia index to enhance its performance.

6. Generate Synthetic Samples

Generate the desired label using the Generate_labels.py script. Specify the arrhythmia index and the number of samples to generate:

bash
Copy code
python Generate_labels.py --label_index arrhythmia_index --num_samples number_of_samples --output_dir path_to_generated_labels
7. Run the Diffusion Model

Run the diffusion model by inserting the generated labels into the inference.py script to generate the corresponding samples:

bash
Copy code
python inference.py --input_labels path_to_generated_labels --output_dir path_to_generated_samples
Combine Train and Validation Sets Again

8. Combine the train and validation sets again using the combine_data.py script (now including both synthetic and real samples):

bash
Copy code
python combine_data.py --input_dir path_to_combined_and_synthetic_data --output_dir path_to_new_combined_data
Retrain the Classifier

9. Retrain the classifier using the train_res50.py script (with the new data containing both synthetic and real samples):

bash
Copy code
python train_res50.py --data_dir path_to_new_combined_data --output_dir path_to_retrained_model
10. Evaluate the Retrained Classifier on the Test Set

Run the retrained classifier on the test set and get the performance report. Compare the results to the baseline report to evaluate the impact of synthetic samples:

bash
Copy code
python test_res50.py --model_dir path_to_retrained_model --test_data_dir path_to_test_data
Important Notes
Test Set Integrity: It is crucial to keep the test set free from synthetic samples to evaluate the quality of the experiment accurately. This ensures the diffusion model's ability to enhance the classifier's performance in diagnosing real-life arrhythmias.
Data Handling: Ensure that the paths to the data directories and model directories are correctly specified in each script.
Contact
For any issues or questions, please contact [ido.mahlab@campus.technion.ac.il].



```bibtex
title = {Enhancing Multilabel Classification of Arrhythmias Using Diffusion-Based Data Augmentation on the PTB-XL Dataset},

keywords = {Cardiology, Electrocardiography, Signal processing, Synthetic data, Diffusion models, Time series, resnet50, classifier},
}

```
Contact
For any issues or questions, please contact [ido.mahlab@campus.technion.ac.il].

Acknowledgments
We would like to acknowledge the following repositories and articles which were instrumental in our research:

SSSD-ECG Repository https://github.com/Idomahlab/SSSD-ECG
Automatic ECG Diagnosis https://github.com/antonior92/automatic-ecg-diagnosis
Generate Figures and Tables Script https://github.com/antonior92/automatic-ecg-diagnosis/blob/tensorflow-v1/generate_figures_and_tables.py