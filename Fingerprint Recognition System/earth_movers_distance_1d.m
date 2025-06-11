function emd = earth_movers_distance_1d(h1, h2)
    % Normalize histograms to probability distributions
    h1 = h1 / sum(h1);
    h2 = h2 / sum(h2);
    
    % Compute cumulative distributions
    c1 = cumsum(h1);
    c2 = cumsum(h2);
    
    % Earth Mover's Distance for 1D histograms
    emd = sum(abs(c1 - c2));
end