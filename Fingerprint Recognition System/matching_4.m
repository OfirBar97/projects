clear; clc;

% --- Load binary fingerprint maps (minutiae maps) ---
A1 = imread('map_of_ones_index_L2.png'); A1 = double(A1 > 0);
A2 = imread('map_of_ones_index_L4.png'); A2 = double(A2 > 0);
A3 = imread('map_of_ones_index_L1.png'); A3 = double(A3 > 0);
A4 = imread('map_of_ones_ofir_1.png'); A4 = double(A4 > 0);

% Store fingerprints
fingerprints = {A1, A2, A3, A4};
labels = {'A1', 'A2', 'A3', 'A4'};
results = [];

% Parameters for orientation-aware matching
sigma_d = 15;         % spatial distance Gaussian std
sigma_theta = pi/12;  % angular diff Gaussian std (15 degrees)
windowSize = 5;       % patch size for orientation smoothing

% --- Main loop: Match each fingerprint against the mean of the other 3 ---
for i = 1:4
    % Reference: the other 3 fingerprints
    ref_indices = setdiff(1:4, i);
    ref_imgs = fingerprints(ref_indices);
    target_img = fingerprints{i};

    % --- Orientation-Aware Matching ---
    ref_points = [];
    for j = 1:3
        [y, x] = find(ref_imgs{j} == 1);
        ori = estimate_orientations(ref_imgs{j}, windowSize);
        idx = sub2ind(size(ref_imgs{j}), y, x);
        angles = ori(idx);
        ref_points = [ref_points; [x, y, angles]];
    end
    ref_points(:,1:2) = ref_points(:,1:2) / 3;  % normalize coordinates across 3 refs

    [yt, xt] = find(target_img == 1);
    oriT = estimate_orientations(target_img, windowSize);
    idxT = sub2ind(size(target_img), yt, xt);
    anglesT = oriT(idxT);
    target_points = [xt, yt, anglesT];

    scores = [];
    for k = 1:size(ref_points,1)
        pt1 = ref_points(k, :);
        dists = sqrt(sum((target_points(:,1:2) - pt1(1:2)).^2, 2));
        angle_diffs = abs(angdiff(pt1(3), target_points(:,3)));

        spatial_score = exp(-(dists.^2) / (2*sigma_d^2));
        angle_score = exp(-(angle_diffs.^2) / (2*sigma_theta^2));
        combined_score = spatial_score .* angle_score;
        scores(end+1) = max(combined_score);
    end
    orientation_similarity = mean(scores) * 100;

    % --- Histogram-Based Matching ---
    dist_maps = cellfun(@shortest_paths_to_nearest_one_diag, ref_imgs, 'UniformOutput', false);
    distT = shortest_paths_to_nearest_one_diag(target_img);

    all_vals = unique([dist_maps{1}(:); dist_maps{2}(:); dist_maps{3}(:); distT(:)]);
    count_total = zeros(size(all_vals));
    for j = 1:3
        count_total = count_total + histc(dist_maps{j}, all_vals);
    end
    mean_count = count_total / 3;
    countT = histc(distT, all_vals);

    intersection = min(mean_count, countT);
    union = max(mean_count, countT);
    valid_idx = (mean_count > 0) & (countT > 0);
    matching_count = sum(intersection(valid_idx));
    total_count = sum(union(valid_idx));
    histogram_similarity = 0;
    if total_count ~= 0
        histogram_similarity = (matching_count / total_count) * 100;
    end

    % --- Combine both similarities ---
    final_similarity = 0.7 * histogram_similarity + 0.3 * orientation_similarity;

    % --- Store results ---
    results(end+1).target = labels{i};
    results(end).histogram_similarity = histogram_similarity;
    results(end).orientation_similarity = orientation_similarity;
    results(end).final_similarity = final_similarity;
end

% --- Display Results ---
fprintf('\n--- Matching Each Sample Against the Mean of the Other 3 ---\n');
for i = 1:length(results)
    fprintf('%s vs. mean of others → Final: %.2f%% (Hist: %.2f%%, Ori: %.2f%%)\n', ...
        results(i).target, results(i).final_similarity, results(i).histogram_similarity, results(i).orientation_similarity);
end

% --- Orientation Estimation Function ---
function orientations = estimate_orientations(binary_img, window_size)
    [Gx, Gy] = gradient(double(binary_img));
    orientations = atan2(Gy, Gx);
    orientations = medfilt2(orientations, [window_size window_size]);
end

% --- Angle Difference Function ---
function diff = angdiff(a1, a2)
    diff = abs(atan2(sin(a1 - a2), cos(a1 - a2)));
end

% --- Distance Map Function ---
function dist = shortest_paths_to_nearest_one_diag(binary_img)
    % Computes shortest diagonal-aware path from each pixel to nearest 1
    dist = bwdist(binary_img, 'quasi-euclidean');
end
