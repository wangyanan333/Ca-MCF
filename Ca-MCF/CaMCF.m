function selfea = CaMCF(data, L, delta1, delta2, k1, k2)
% Inputs:
%   data   - (N x P) matrix.
%   L      - Number of labels.
%   delta1 - Feature-category correlation threshold.
%   delta2 - Category-category correlation threshold.
%
% Output:
%   selfea - [Vector] Global feature subset indices.

    addpath('common');
    [N, p] = size(data);
    num_feats = p - L;
    
    label_indices = (num_feats + 1) : p;
    feature_indices = 1 : num_feats;
    
    global_selected_features = [];
    
    fprintf('Starting Ca-MCF (Strict Class-Specific Mode)...\n');
    
    for i = 1:L
        current_lbl_idx = label_indices(i);
        unique_classes = unique(data(:, current_lbl_idx));
        
        for c_idx = 1:length(unique_classes)
            target_val = unique_classes(c_idx);
            
            % Construct class-specific binary vector
            target_vec = double(data(:, current_lbl_idx) == target_val);
            
            % Skip categories with insufficient samples
            if sum(target_vec) < 5, continue; end
            Matrix_Ric = [];
            
            other_lbl_indices = setdiff(label_indices, current_lbl_idx);
            
            % ----------------------------------------------------
            % Phase 1: Label-Category Dependency Modeling
            % ----------------------------------------------------
            cand_label_classes = []; 
            
            for j_lbl = other_lbl_indices
                u_vals = unique(data(:, j_lbl));
                for d_idx = 1:length(u_vals)
                    other_val = u_vals(d_idx);
                    other_vec = double(data(:, j_lbl) == other_val);
                    val = mi(target_vec, other_vec);
                    
                    if val > delta2
                        cand_label_classes = [cand_label_classes; j_lbl, other_val, val];
                    end
                end
            end
            
            if ~isempty(cand_label_classes)
                % Sort candidates by mutual information scores
                cand_label_classes = sortrows(cand_label_classes, -3); 
                temp_cand_vecs = zeros(N, size(cand_label_classes, 1));
                for k = 1:size(cand_label_classes, 1)
                     l_idx = cand_label_classes(k, 1);
                     v_val = cand_label_classes(k, 2);
                     temp_cand_vecs(:, k) = double(data(:, l_idx) == v_val);
                end
                
                % Prune redundant dependencies via conditional DCSMI tests
                selected_indices = [];
                for k = 1:size(temp_cand_vecs, 2)
                    current_cand_vec = temp_cand_vecs(:, k);
                    
                    if isempty(selected_indices)
                        selected_indices = [selected_indices, k];
                    else
                        existing_data = temp_cand_vecs(:, selected_indices);
                        val = DCSMI(target_vec, current_cand_vec, existing_data);
                        
                        if val > delta2
                            selected_indices = [selected_indices, k];
                        end
                    end
                end
                Matrix_Ric = temp_cand_vecs(:, selected_indices);
            end
            
            % ----------------------------------------------------
            % Phase 2: Local Structural Discovery
            % ----------------------------------------------------
            
            % 2.1 PC Discovery (Find Parents and Children)
            Cand_PC = [];
            for f_idx = feature_indices
                val = mi(target_vec, data(:, f_idx));
                if val > delta1
                    Cand_PC = [Cand_PC; f_idx, val];
                end
            end
            
            PC_indices = [];
            if ~isempty(Cand_PC)
                Cand_PC = sortrows(Cand_PC, -2);
                num_keep_k1 = ceil(size(Cand_PC, 1) * k1);
                Cand_PC = Cand_PC(1:num_keep_k1, :);
                current_PC = Cand_PC(:, 1)';
                
                final_PC = current_PC;
                for f_k = current_PC
                    S = setdiff(final_PC, f_k);
                    if isempty(S)
                        Data_Cond = Matrix_Ric;
                    else
                        Data_Cond = [data(:, S), Matrix_Ric];
                    end
                    
                    if isempty(Data_Cond)
                        val = mi(target_vec, data(:, f_k));
                    else
                        val = SCSMI(target_vec, data(:, f_k), Data_Cond);
                    end
                    
                    % Prune features that become independent given the conditioning set
                    if val <= delta1
                        final_PC(final_PC == f_k) = [];
                    end
                end
                PC_indices = final_PC;
            end
            
            % 2.2 Spouse Discovery (Identify V-structures)
            SP_indices = [];
            candidates_Z = setdiff(feature_indices, PC_indices);
            
            for x_node = PC_indices
                for z_node = candidates_Z
                    if isempty(Matrix_Ric)
                        val1 = mi(target_vec, data(:, z_node));
                    else
                        val1 = SCSMI(target_vec, data(:, z_node), Matrix_Ric);
                    end
                    
                    if val1 <= delta1
                        if isempty(Matrix_Ric)
                            Cond_XR = data(:, x_node);
                        else
                            Cond_XR = [data(:, x_node), Matrix_Ric];
                        end
                        
                        val2 = SCSMI(target_vec, data(:, z_node), Cond_XR);
                        
                        if val2 > delta1
                            SP_indices = [SP_indices, z_node];
                        end
                    end
                end
            end
            SP_indices = unique(SP_indices);
            CMB_indices = unique([PC_indices, SP_indices]);
            
            % ----------------------------------------------------
            % Phase 3: Class-Aware Feature Recovery
            % ----------------------------------------------------
            
            if ~isempty(Matrix_Ric)
                num_ric = size(Matrix_Ric, 2);
                keep_ric_mask = true(1, num_ric); 
                
                F_miss_candidates = setdiff(feature_indices, CMB_indices);
                
                for f_miss = F_miss_candidates
                    % Heuristic filter for candidates
                    if mi(target_vec, data(:, f_miss)) < delta1/2, continue; end
                    
                    replaced_flag = false;
                    for r_i = 1:num_ric
                        if ~keep_ric_mask(r_i), continue; end
                        
                        Y_block_vec = Matrix_Ric(:, r_i); 
                        other_ric_idx = setdiff(1:num_ric, r_i);
                        Data_S_base = [data(:, PC_indices), Matrix_Ric(:, other_ric_idx)];
                        
                        if isempty(Data_S_base)
                            val_feat = mi(target_vec, data(:, f_miss));
                            val_block = mi(target_vec, Y_block_vec);
                        else
                            val_feat = SCSMI(target_vec, data(:, f_miss), Data_S_base);
                            val_block = DCSMI(target_vec, Y_block_vec, Data_S_base);
                        end
                        
                        % Recover features with higher conditional explanatory power
                        if val_feat > val_block
                            CMB_indices = [CMB_indices, f_miss];
                            keep_ric_mask(r_i) = false; 
                            replaced_flag = true;
                            break; 
                        end
                    end
                    if replaced_flag, break; end 
                end
            end
            
            % ----------------------------------------------------
            % Phase 4: Symmetry Check and Redundancy Removal
            % ----------------------------------------------------
            
            % 4.1 Causal Symmetry Check (Target Class <-> Feature)
            current_CMB = CMB_indices;
            scores = zeros(length(current_CMB), 1);
            for k = 1:length(current_CMB)
                elem = current_CMB(k);
                scores(k) = mi(data(:, elem), target_vec); 
            end
            [~, sort_idx] = sort(scores, 'descend');
            sorted_CMB = current_CMB(sort_idx);
            
            num_keep = ceil(length(sorted_CMB) * k2);
            top_CMB = sorted_CMB(1:num_keep);
            
            final_CMB_symmetry = [];
            for elem = top_CMB
                if mi(data(:, elem), target_vec) > delta1
                    final_CMB_symmetry = [final_CMB_symmetry, elem];
                end
            end
            CMB_indices = final_CMB_symmetry;
            
            % 4.2 Cross-Dimensional Redundancy Removal
            final_MB_clean = CMB_indices;
            for f_k = CMB_indices
                val_current = mi(target_vec, data(:, f_k));
                is_redundant = false;
                for j_lbl = other_lbl_indices
                    u_vals_other = unique(data(:, j_lbl));
                    for d_idx = 1:length(u_vals_other)
                        other_val = u_vals_other(d_idx);
                        other_vec = double(data(:, j_lbl) == other_val);
                        
                        % Feature <-> Other Class association check
                        val_other = mi(other_vec, data(:, f_k)); 
                        
                        if val_other > (val_current * 1.2)
                            is_redundant = true;
                            break;
                        end
                    end
                    if is_redundant, break; end
                end
                
                if is_redundant
                    final_MB_clean(final_MB_clean == f_k) = [];
                end
            end
            
            global_selected_features = [global_selected_features, final_MB_clean];
            
        end 
    end
    
    % Final set of unique features across all labels and categories
    selfea = unique(global_selected_features);
    fprintf('Ca-MCF Completed. Selected %d unique features.\n', length(selfea));
end