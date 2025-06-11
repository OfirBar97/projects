function similarity = histogram_similarity(h1, h2, method)
    % Normalize histograms
    if sum(h1) > 0
        h1_norm = h1 / sum(h1);
    else
        h1_norm = h1;
    end
    
    if sum(h2) > 0
        h2_norm = h2 / sum(h2);
    else
        h2_norm = h2;
    end
    
    switch method
        case 'intersection'
            % Histogram intersection
            similarity = sum(min(h1_norm, h2_norm));
        case 'chi_square'
            % Chi-square distance (converted to similarity)
            chi_sq = sum((h1_norm - h2_norm).^2 ./ (h1_norm + h2_norm + eps));
            similarity = 1 / (1 + chi_sq);
        case 'bhattacharyya'
            % Bhattacharyya coefficient
            similarity = sum(sqrt(h1_norm .* h2_norm));
        case 'cosine'
            % Cosine similarity
            similarity = sum(h1_norm .* h2_norm) / (sqrt(sum(h1_norm.^2)) * sqrt(sum(h2_norm.^2)) + eps);
        case 'earth_mover'
            % Simple approximation of Earth Mover's Distance for 1D histograms
            cdf1 = cumsum(h1_norm);
            cdf2 = cumsum(h2_norm);
            emd = sum(abs(cdf1 - cdf2));
            similarity = 1 / (1 + emd);
        case 'jaccard'
            % Jaccard similarity
            intersection = sum(min(h1_norm, h2_norm));
            union = sum(max(h1_norm, h2_norm));
            similarity = intersection / (union + eps);
        case 'euclidean'
            % Euclidean distance (converted to similarity)
            dist = sqrt(sum((h1_norm - h2_norm).^2));
            similarity = 1 / (1 + dist);
        case 'chebychev'
            chebyshev = max(abs(h1_norm - h2_norm));
            similarity = 1 / (1 + chebyshev);
        case 'manhattan'
            % Manhattan (L1) distance (converted to similarity)
            dist = sum(abs(h1_norm - h2_norm));
            similarity = 1 / (1 + dist);
        otherwise
            error('Unknown similarity method');
    end
end