function loss = calculate_hamming_loss(Y_true, Y_pred)
% calculate_hamming_loss: Computes the Hamming Loss for multi-label classification.
% The Hamming loss is the fraction of labels that are incorrectly predicted.
%
% Inputs:
%   Y_true - A P x Q binary matrix of true labels (P instances, Q labels).
%   Y_pred - A P x Q binary matrix of predicted labels.
%
% Output:
%   loss   - The calculated Hamming Loss (a scalar value between 0 and 1).

    % Ensure inputs are binary (0s and 1s) and of the same size.
    if ~isequal(size(Y_true), size(Y_pred))
        error('True labels and predicted labels matrices must be the same size.');
    end
    
    [p, q] = size(Y_true);
    
    % The symmetric difference |Qi Δ Yi| for binary vectors is the number
    % of positions where they differ, which can be computed with xor.
    % We sum the number of differing elements over all instances and labels.
    num_misclassified_labels = sum(sum(xor(Y_true, Y_pred)));
    
    % Total number of labels is p * q.
    total_labels = p * q;
    
    % Hamming Loss is the average misclassification rate.
    loss = num_misclassified_labels / total_labels;
end