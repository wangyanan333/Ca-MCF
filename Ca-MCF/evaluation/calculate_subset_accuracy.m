function accuracy = calculate_subset_accuracy(Y_true, Y_pred)
% calculate_subset_accuracy: Computes the proportion of instances where
% the predicted label set is an exact match to the true label set.

    % Compare each row for an exact match
    exact_matches = all(Y_true == Y_pred, 2);
    
    % The accuracy is the mean of the indicator vector
    accuracy = mean(exact_matches);
end