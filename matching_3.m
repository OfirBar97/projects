clear; clc;

% --- Load binary fingerprint maps (minutiae maps) ---
A1 = imread('map_of_ones_ofir_1.png'); A1 = double(A1 > 0);
A2 = imread('map_of_ones_ofir_2.png'); A2 = double(A2 > 0);
A3 = imread('map_of_ones_ofir_4.png'); A3 = double(A3 > 0);
A4 = imread('map_of_ones_ofir_5.png'); A4 = double(A4 > 0);  % Test fingerprint

% Store fingerprints
fingerprints = {A1, A2, A3, A4};
labels = {'A1', 'A2', 'A3', 'A4'};
results = [];

% Parameters for orientation-aware matching
sigma_d = 15;         % spatial distance Gaussian std
sigma_theta = pi/12;  % angular diff Gaussian std (15 degrees)
windowSize = 5;      % patch size for orientation

% Loop to test each permutation of A1-A3 replaced by A4
for i = 1:3
    refSet = fingerprints;
    refSet{i} = fingerprints{4};  % Replace one of A1-A3 with A4

    % --- Orientation-Aware Matching ---
    % Collect minutiae points and orientations
    ref_points = [];
    for j = 1:3
        [y, x] = find(refSet{j} == 1);
        ori = estimate_orientations(refSet{j}, windowSize);
        idx = sub2ind(size(refSet{j}), y, x);
        angles = ori(idx);
        ref_points = [ref_points; [x, y, angles]];
    end
    ref_points(:,1:2) = ref_points(:,1:2) / 3;  % average x, y (roughly)

    % Target fingerprint points
    target_img = fingerprints{4};
    [yt, xt] = find(target_img == 1);
    oriT = estimate_orientations(target_img, windowSize);
    idxT = sub2ind(size(target_img), yt, xt);
    anglesT = oriT(idxT);
    target_points = [xt, yt, anglesT];

    % --- Match with orientation-aware scoring ---
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
    dist1 = shortest_paths_to_nearest_one_diag(refSet{1});
    dist2 = shortest_paths_to_nearest_one_diag(refSet{2});
    dist3 = shortest_paths_to_nearest_one_diag(refSet{3});
    distB = shortest_paths_to_nearest_one_diag(fingerprints{4});

    all_vals = unique([dist1(:); dist2(:); dist3(:); distB(:)]);
    count1 = histc(dist1, all_vals);
    count2 = histc(dist2, all_vals);
    count3 = histc(dist3, all_vals);
    mean_count = (count1 + count2 + count3) / 3;
    countB = histc(distB, all_vals);

    intersection = min(mean_count, countB);
    union = max(mean_count, countB);
    valid_idx = (mean_count > 0) & (countB > 0);
    matching_count = sum(intersection(valid_idx));
    total_count = sum(union(valid_idx));
    histogram_similarity = 0;
    if total_count ~= 0
        histogram_similarity = (matching_count / total_count) * 100;
    end

    % Combine both similarities (weighted average)
    final_similarity = 0.7 * histogram_similarity + 0.3 * orientation_similarity;

    % Store result
    results(end+1).replaced = labels{i};
    results(end).histogram_similarity = histogram_similarity;
    results(end).orientation_similarity = orientation_similarity;
    results(end).final_similarity = final_similarity;
end

% --- Display Results ---
fprintf('\n--- Combined Matching Results (Histogram + Orientation) ---\n');
if final_similarity >70
    fprintf('\n--- Fingerprints Are Matched ---\n');
else
    fprintf('\n--- Fingerprints Are *Not* Matched ---\n');
end
for i = 1:length(results)
    fprintf('Replacing %s with A4 → Final Similarity: %.2f%% (Hist: %.2f%%, Ori: %.2f%%)\n', ...
        results(i).replaced, results(i).final_similarity, results(i).histogram_similarity, results(i).orientation_similarity);
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