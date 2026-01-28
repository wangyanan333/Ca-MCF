function val = DCSMI(target_category_Cic, category_Cjd, conditioning_set_S)
% DCSMI: Dual-Category Specific Mutual Information
    h_YZ = h_multivariate([category_Cjd, conditioning_set_S]);
    h_XZ = h_multivariate([target_category_Cic, conditioning_set_S]);
    h_Z  = h(conditioning_set_S);
    h_YXZ = h_multivariate([category_Cjd, target_category_Cic, conditioning_set_S]);
    
    val = h_YZ + h_XZ - h_Z - h_YXZ;
end
function H = h_multivariate(data)
    [~, ~, uidx] = unique(data, 'rows');
    p = accumarray(uidx, 1) / size(data, 1);
    H = -sum(p .* log2(p));
end