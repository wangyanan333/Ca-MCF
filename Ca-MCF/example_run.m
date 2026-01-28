function example_run()
% -------------------------------------------------------------------------
% 1. Setup and Initialization
% -------------------------------------------------------------------------
clear;               
clc;                  
close all;           
addpath('common', 'classifiers', 'evaluation'); 

% -------------------------------------------------------------------------
% 2. Configuration Parameters
% -------------------------------------------------------------------------
DATASET_NAME = 'Flags';          
L_original = 7;                  
k_knn = 10;                     
delta1 = 0.05;                   % Threshold for feature-label correlation
delta2 = 0.1;                    % Threshold for label-label correlation
k1 = 0.7;                        % Top-k ratio for Phase 1 (Discovery)
k2 = 1.0;                        % Top-k ratio for Phase 3 (Symmetry)

disp('--- Configuration (Ca-MCF) ---');
disp(['Delta1: ', num2str(delta1), ', Delta2: ', num2str(delta2), ...
      ', k1: ', num2str(k1), ', k2: ', num2str(k2), ', k-NN: ', num2str(k_knn)]);
disp('------------------------------');

% -------------------------------------------------------------------------
% 3. Data Loading and Initial Partition
% -------------------------------------------------------------------------
disp('1. Loading training and testing data...');
train_dataset = load(['data/', DATASET_NAME, '/', DATASET_NAME, '-train.mat']);
test_dataset = load(['data/', DATASET_NAME, '/', DATASET_NAME, '-test.mat']);
train_data_raw = double(train_dataset.train); 
test_data_raw = double(test_dataset.test);    
disp('Data loaded.');

disp('2. Partitioning data and setting labels...');
[~, p_raw] = size(train_data_raw);
num_features = p_raw - L_original;

% Extract training/testing features and labels
X_train = train_data_raw(:, 1:num_features);
Y_train_raw = train_data_raw(:, (num_features + 1):end);
X_test = test_data_raw(:, 1:num_features);
Y_test_raw = test_data_raw(:, (num_features + 1):end);

% Label preparation
Y_train_binarized = Y_train_raw; 
Y_test_binarized = Y_test_raw;
L_new = size(Y_train_binarized, 2);

disp(['Labels kept at L_original: ', num2str(L_original), ...
      '. New binary labels: ', num2str(L_new), ' (DIMENSION PRESERVED)']); 

% Concatenate matrix for feature selection
train_data_processed = [X_train, Y_train_binarized];

% -------------------------------------------------------------------------
% 4. Feature Selection
% -------------------------------------------------------------------------
disp('3. Running Ca-MCF algorithm...');
% Call CaMCF algorithm for feature selection
selected_features = CaMCF(train_data_processed, L_new, delta1, delta2, k1, k2); 
num_selected_features = length(selected_features);
disp(['Ca-MCF completed. ', num2str(num_selected_features), ' features selected.']);

% -------------------------------------------------------------------------
% 5. ML-kNN Prediction
% -------------------------------------------------------------------------
disp('4. Training ML-kNN and predicting...');
% Extract only the selected feature subset
X_train_final = X_train(:, selected_features); 
X_test_final = X_test(:, selected_features);   

[Y_pred, F_pred] = ML_KNN(X_train_final, Y_train_binarized, X_test_final, k_knn);
disp('Prediction completed.');

% -------------------------------------------------------------------------
% 6. Evaluation and Display
% -------------------------------------------------------------------------
disp('5. Calculating performance metrics...');
ham_loss = calculate_hamming_loss(Y_test_binarized, Y_pred);
sub_acc = calculate_subset_accuracy(Y_test_binarized, Y_pred);
[macro_f1, micro_f1] = calculate_f_scores(Y_test_binarized, Y_pred);
[avg_prec, coverage, rank_loss] = calculate_ranking_metrics(Y_test_binarized, F_pred);

disp(' ');
disp('==================== FINAL RESULTS (Ca-MCF) ====================');
MetricNames = ["Hamming Loss (↓)"; "Subset Accuracy (↑)"; "Average Precision (↑)"; ...
               "Coverage (↓)"; "Ranking Loss (↓)"; "Macro-F1 (↑)"; "Micro-F1 (↑)"];
Values = [ham_loss; sub_acc; avg_prec; coverage; rank_loss; macro_f1; micro_f1];
results = table(MetricNames, Values);
disp(results);             
disp('==============================================================');

% -------------------------------------------------------------------------
% 7. Variable Summary
% -------------------------------------------------------------------------
fprintf('\n');
disp('Process finished. Results are stored in memory:');
disp(['- Selected Feature Indices: selected_features (Count: ', num2str(num_selected_features), ')']);
disp(['- Final Predictions: Y_pred, Size: ', mat2str(size(Y_pred))]);
disp(['- Confidence Scores: F_pred, Size: ', mat2str(size(F_pred))]);

end