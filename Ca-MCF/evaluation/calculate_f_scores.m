function [macro_f1, micro_f1] = calculate_f_scores(Y_true, Y_pred)
% calculate_f_scores: Computes both Macro-averaged and Micro-averaged F1 scores.

    % Calculate TP, FP, FN for each label (column-wise)
    TP = sum(Y_true == 1 & Y_pred == 1, 1);
    FP = sum(Y_true == 0 & Y_pred == 1, 1);
    FN = sum(Y_true == 1 & Y_pred == 0, 1);

    % --- Macro-F1: Average of F1 scores per label ---
    % Calculate F1 for each label, handle division by zero
    f1_per_label = (2 * TP) ./ (2 * TP + FP + FN);
    f1_per_label(isnan(f1_per_label)) = 0; % Set NaN results (0/0) to 0
    macro_f1 = mean(f1_per_label);

    % --- Micro-F1: F1 score from aggregated counts ---
    total_tp = sum(TP);
    total_fp = sum(FP);
    total_fn = sum(FN);
    
    % Handle division by zero for the overall score
    if (total_tp + total_fp + total_fn) == 0
        micro_f1 = 0;
    else
        micro_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn);
    end
end