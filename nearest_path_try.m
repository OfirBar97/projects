clear; clc;

% ==== Read and process fingerprint 1 (ofir_7_1) ====
ridge1 = double(imread('map_of_ridge_ofir_7_4.png') > 0);
bif1   = double(imread('map_of_bif_ofir_7_4.png') > 0);
img1   = imread('ofir_7_1.bmp');

dist_ridge1 = shortest_paths_to_nearest_one_diag(ridge1);
dist_bif1   = shortest_paths_to_nearest_one_diag(bif1);

% ==== Read and process fingerprint 2 (ofir_7_2) ====
ridge2 = double(imread('map_of_ridge_ofir_7_1.png') > 0);
bif2   = double(imread('map_of_bif_ofir_7_1.png') > 0);
img2   = imread('ofir_7_2.bmp');

dist_ridge2 = shortest_paths_to_nearest_one_diag(ridge2);
dist_bif2   = shortest_paths_to_nearest_one_diag(bif2);

% ==== Compare Ridge Distances using bin-wise min/max ratio ====
all_vals_ridge = unique([dist_ridge1(:); dist_ridge2(:)]);
count_ridge1 = histc(dist_ridge1(:), all_vals_ridge);
count_ridge2 = histc(dist_ridge2(:), all_vals_ridge);

max_ridge = max(count_ridge1, count_ridge2);
valid_bins_ridge = max_ridge > 0;
bin_sim_ridge = min(count_ridge1(valid_bins_ridge), count_ridge2(valid_bins_ridge)) ./ max_ridge(valid_bins_ridge);
match_ridge = 100 * mean(bin_sim_ridge);

% ==== Compare Combined Ridge + Bifurcation Distances using bin-wise min/max ratio ====
combined1 = [dist_ridge1(:); dist_bif1(:)];
combined2 = [dist_ridge2(:); dist_bif2(:)];

all_vals_comb = unique([combined1; combined2]);
count_comb1 = histc(combined1, all_vals_comb);
count_comb2 = histc(combined2, all_vals_comb);

max_comb = max(count_comb1, count_comb2);
valid_bins_comb = max_comb > 0;
bin_sim_comb = min(count_comb1(valid_bins_comb), count_comb2(valid_bins_comb)) ./ max_comb(valid_bins_comb);
match_comb = 100 * mean(bin_sim_comb);

% ==== Combined Similarity Score ====
similarity = 0.5 * match_ridge + 0.5 * match_comb;


% ==== Determine match status ====
if similarity > 45
    state = 'Are Matched';
else
    state = 'Are Not Matched';
end

% ==== Display results ====
disp(['Ridge Match %: ', num2str(match_ridge, '%.2f'), '%']);
disp(['Combined Ridge+Bif Match %: ', num2str(match_comb, '%.2f'), '%']);
disp(['Overall Match %: ', num2str(similarity, '%.2f'), '% — ', state]);

% ==== Visualization ====
figure;
tiledlayout(5, 2, 'TileSpacing', 'compact');

% Row 1: histogram comparison (Ridge)
nexttile([1 2])
bar(all_vals_ridge, [count_ridge1, count_ridge2], 'grouped');
xlabel('Distance Values');
ylabel('Occurrences');
legend('Ridge 7\_1', 'Ridge 7\_2');
title('Histogram: Ridge Distances');

% Row 3: histogram comparison (Combined)
nexttile([1 2])
bar(all_vals_comb, [count_comb1, count_comb2], 'grouped');
xlabel('Distance Values');
ylabel('Occurrences');
legend('Combined 7\_1', 'Combined 7\_2');
title('Histogram: Ridge + Bifurcation Distances');

% Row 4: fingerprint images
nexttile
imshow(img1);
title('Fingerprint: Ofir 7\_1');

nexttile
imshow(img2);
title('Fingerprint: Ofir 7\_2');

% Shared title
sgtitle(sprintf('Fingerprint Match: %.2f%% — %s', similarity, state));

