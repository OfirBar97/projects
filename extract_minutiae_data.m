function final_data = extract_minutiae_data(map_bif, map_ridge, map_size)

    % Step 1: Create binary maps
    map_of_ones_bif    = create_map_from_coordinates(map_bif, map_size);
    map_of_ones_ridges = create_map_from_coordinates(map_ridge, map_size);

    % Step 2: Compute distance maps
    dist_ridge   = shortest_paths_to_nearest_one_diag(map_of_ones_ridges);
    dist_bif     = shortest_paths_to_nearest_one_diag(map_of_ones_bif);
    dist_combined = shortest_paths_to_nearest_one_diag(map_of_ones_ridges + map_of_ones_bif);

    [H, W] = size(dist_ridge);  % Assume all distance maps are same size

    % Step 3: Extract and filter ridge coordinates
    [y_rid, x_rid] = find(map_of_ones_ridges);
    valid_ridges = x_rid >= 1 & x_rid <= W & y_rid >= 1 & y_rid <= H;
    x_rid = x_rid(valid_ridges);
    y_rid = y_rid(valid_ridges);
    ridge_idx = sub2ind([H, W], y_rid, x_rid);

    % Step 4: Extract and filter bifurcation coordinates
    [y_bif, x_bif] = find(map_of_ones_bif);
    valid_bif = x_bif >= 1 & x_bif <= W & y_bif >= 1 & y_bif <= H;
    x_bif = x_bif(valid_bif);
    y_bif = y_bif(valid_bif);
    bif_idx = sub2ind([H, W], y_bif, x_bif);

    % Step 5: Construct data
    ridge_data = [x_rid, y_rid, ones(numel(x_rid), 1), ...
                  dist_ridge(ridge_idx), dist_bif(ridge_idx), dist_combined(ridge_idx)];

    bif_data = [x_bif, y_bif, 3 * ones(numel(x_bif), 1), ...
                dist_ridge(bif_idx), dist_bif(bif_idx), dist_combined(bif_idx)];

    fprintf("Number of ridge coordinates (1s in map_of_ones_ridges): %d\n", nnz(ridge_data));
fprintf("Number of bifurcation coordinates (1s in map_of_ones_bif): %d\n", nnz(bif_data));

    % Step 6: Combine
    final_data = [ridge_data; bif_data];
    disp(final_data)

end
