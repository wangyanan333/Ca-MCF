function [avg_precision, coverage, ranking_loss] = calculate_ranking_metrics(Y_true, F_pred)
% calculate_ranking_metrics: Computes Average Precision, Coverage, and Ranking Loss.
%
% UPDATED: Coverage is now NORMALIZED to the [0, 1] range to match
% common reporting standards in academic papers.

    [p, num_labels] = size(Y_true);
    
    % Convert scores to ranks. A higher score gets a lower rank number (e.g., rank 1).
    [~, sorted_indices] = sort(F_pred, 2, 'descend');
    [~, ranks] = sort(sorted_indices, 2);

    total_ap = 0;
    total_cov_standard = 0; % We will calculate the standard coverage first
    total_rl = 0;

    for i = 1:p
        true_labels_idx = find(Y_true(i, :));
        false_labels_idx = find(~Y_true(i, :));
        num_true_labels = length(true_labels_idx);
        num_false_labels = length(false_labels_idx);
        
        if num_true_labels == 0, continue; end % Skip if no true labels
        
        ranks_of_true_labels = ranks(i, true_labels_idx);
        
        % --- Average Precision ---
        instance_ap = 0;
        for r_k = ranks_of_true_labels
            num_better_or_equal = sum(ranks_of_true_labels <= r_k);
            instance_ap = instance_ap + (num_better_or_equal / r_k);
        end
        total_ap = total_ap + (instance_ap / num_true_labels);

        % --- Standard Coverage ---
        max_rank = max(ranks_of_true_labels);
        total_cov_standard = total_cov_standard + (max_rank - 1);
        
        % --- Ranking Loss ---
        if num_false_labels > 0
            misordered_pairs = 0;
            for true_idx = true_labels_idx
                for false_idx = false_labels_idx
                    if ranks(i, true_idx) > ranks(i, false_idx)
                        misordered_pairs = misordered_pairs + 1;
                    end
                end
            end
            total_rl = total_rl + (misordered_pairs / (num_true_labels * num_false_labels));
        end
    end
    
    % --- Final Metric Calculations ---
    avg_precision = total_ap / p;
    ranking_loss = total_rl / p;
    
    % --- NORMALIZATION of Coverage ---
    % The standard coverage is the average of (max_rank - 1)
    coverage_standard = total_cov_standard / p;
    
    % Normalize by dividing by the maximum possible value, which is (num_labels - 1)
    if num_labels > 1
        coverage = coverage_standard / (num_labels - 1);
    else
        coverage = 0; % If only one label, coverage is trivially 0
    end
end