function [Y_pred, F_pred] = ML_KNN(X_train, Y_train, X_test, k)
% ML_KNN: Multi-Label k-Nearest Neighbor (Standard Implementation)
% Based on Zhang & Zhou (2007)
% 
% Inputs:
%   X_train: Training features (N_train x M)
%   Y_train: Training labels (N_train x L) - Binary (0/1)
%   X_test:  Test features (N_test x M)
%   k:       Number of neighbors
%
% Outputs:
%   Y_pred: Predicted binary labels (N_test x L)
%   F_pred: Predicted confidence scores (N_test x L)

    [num_train, num_labels] = size(Y_train);
    num_test = size(X_test, 1);
    
    s = 1; 
    Ph1 = (s + sum(Y_train, 1)) / (s * 2 + num_train); 
    Ph0 = 1 - Ph1;                                     
   
    Count_H1 = zeros(k + 1, num_labels); 
    Count_H0 = zeros(k + 1, num_labels);
    
    dist_type = 'hamming'; 
    idx_train = knnsearch(X_train, X_train, 'K', k+1, 'Distance', dist_type); 
    
    idx_train = idx_train(:, 2:end); 
    
    for i = 1:num_train
        neighbor_labels = Y_train(idx_train(i,:), :);
        temp_C = sum(neighbor_labels, 1); 
        
        for l = 1:num_labels
            c = temp_C(l); 
            if Y_train(i, l) == 1
                Count_H1(c + 1, l) = Count_H1(c + 1, l) + 1;
            else
                Count_H0(c + 1, l) = Count_H0(c + 1, l) + 1;
            end
        end
    end

    total_H1 = sum(Y_train, 1);
    total_H0 = num_train - total_H1;
    
    Cond_Prob_True = (s + Count_H1) ./ (s * (k + 1) + repmat(total_H1, k+1, 1));
    Cond_Prob_False = (s + Count_H0) ./ (s * (k + 1) + repmat(total_H0, k+1, 1));

    Y_pred = zeros(num_test, num_labels);
    F_pred = zeros(num_test, num_labels);
    
    idx_test = knnsearch(X_train, X_test, 'K', k, 'Distance', dist_type);
    
    for i = 1:num_test
        neighbor_labels = Y_train(idx_test(i, :), :);
        C_test = sum(neighbor_labels, 1); 
        
        for l = 1:num_labels
            c = C_test(l);
            prob_y1_given_c = Ph1(l) * Cond_Prob_True(c + 1, l);
            prob_y0_given_c = Ph0(l) * Cond_Prob_False(c + 1, l);

            if (prob_y1_given_c + prob_y0_given_c) == 0
                F_pred(i, l) = Ph1(l);
            else
                F_pred(i, l) = prob_y1_given_c / (prob_y1_given_c + prob_y0_given_c);
            end

            Y_pred(i, l) = F_pred(i, l) > 0.5;
        end
    end
end