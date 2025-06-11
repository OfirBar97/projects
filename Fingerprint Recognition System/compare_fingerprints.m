function [similarity, match_ridge, match_comb, w_ridge, w_comb] = compare_fingerprints(ridge1, bif1, ridge2, bif2, dist_ridge1, dist_bif1, dist_ridge2, dist_bif2, similarity_method)
    % Ridge Histogram
    all_vals_ridge = unique([dist_ridge1(:); dist_ridge2(:)]);
    count_ridge1 = histc(dist_ridge1(:), all_vals_ridge);
    count_ridge2 = histc(dist_ridge2(:), all_vals_ridge);
    match_ridge = histogram_similarity(count_ridge1, count_ridge2, similarity_method);
    
    % Combined Histogram
    combined1 = [dist_ridge1(:); dist_bif1(:)];
    combined2 = [dist_ridge2(:); dist_bif2(:)];
    all_vals_comb = unique([combined1; combined2]);
    count_comb1 = histc(combined1, all_vals_comb);
    count_comb2 = histc(combined2, all_vals_comb);
    match_comb = histogram_similarity(count_comb1, count_comb2, similarity_method);
    
    % Feature-level fusion with adaptive weights
    ridge_entropy = -sum((count_ridge1/sum(count_ridge1) + eps) .* log2(count_ridge1/sum(count_ridge1) + eps));
    comb_entropy = -sum((count_comb1/sum(count_comb1) + eps) .* log2(count_comb1/sum(count_comb1) + eps));
    
    total_entropy = ridge_entropy + comb_entropy;
    if total_entropy > 0
        w_ridge = ridge_entropy / total_entropy;
        w_comb = comb_entropy / total_entropy;
    else
        % Default weights if entropy calculation fails
        w_ridge = 0.5;
        w_comb = 0.5;
    end
    
    % Apply adaptive weights
    similarity = w_ridge * match_ridge + w_comb * match_comb;
end