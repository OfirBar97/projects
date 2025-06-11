clear; clc;

% ==== Read and process fingerprint 1 ====
ridge1 = double(imread('map_of_ridge_ofir_7_3.png') > 0);
bif1   = double(imread('map_of_bif_ofir_7_3.png') > 0);
img1   = imread('ofir_7_3.bmp');

dist_ridge1 = shortest_paths_to_nearest_one_diag(ridge1);
dist_bif1   = shortest_paths_to_nearest_one_diag(bif1);

% ==== Read and process fingerprint 2 ====
ridge2 = double(imread('map_of_ridge_ofir_7_2.png') > 0);
bif2   = double(imread('map_of_bif_ofir_7_2.png') > 0);
img2   = imread('ofir_7_1.bmp');

dist_ridge2 = shortest_paths_to_nearest_one_diag(ridge2);
dist_bif2   = shortest_paths_to_nearest_one_diag(bif2);

% ==== Similarity Function Based on Min(I_j, M_j) ====
similarity_min = @(h1, h2) sum(min(h1, h2)) / sum(h2);




  % Modified to use your formula

% ==== Ridge Histogram ====
all_vals_ridge = unique([dist_ridge1(:); dist_ridge2(:)]);
count_ridge1 = histc(dist_ridge1(:), all_vals_ridge);
count_ridge2 = histc(dist_ridge2(:), all_vals_ridge);
match_ridge = similarity_min(count_ridge1, count_ridge2);

% ==== Bifurcation Histogram ====
all_vals_bif = unique([dist_bif1(:); dist_bif2(:)]);
count_bif1 = histc(dist_bif1(:), all_vals_bif);
count_bif2 = histc(dist_bif2(:), all_vals_bif);
match_bif = similarity_min(count_bif1, count_bif2);

% ==== Combined Histogram ====
combined1 = [dist_ridge1(:); dist_bif1(:)];
combined2 = [dist_ridge2(:); dist_bif2(:)];
all_vals_comb = unique([combined1; combined2]);
count_comb1 = histc(combined1, all_vals_comb);
count_comb2 = histc(combined2, all_vals_comb);
match_comb = similarity_min(count_comb1, count_comb2);

% ==== Weighted Overall Similarity ====
similarity = 0.4 * match_ridge + 0.1 * match_bif + 0.5 * match_comb;

% ==== Match Decision ====
if similarity > 0.45  % Assuming a threshold of 45% for matching
    state = 'Are Matched';
else
    state = 'Are Not Matched';
end

% ==== Display Results ====
disp(['Ridge Similarity: ', num2str(match_ridge, '%.2f')]);
disp(['Bifurcation Similarity: ', num2str(match_bif, '%.2f')]);
disp(['Combined Similarity: ', num2str(match_comb, '%.2f')]);
disp(['Overall Match %: ', num2str(similarity, '%.2f'), ' — ', state]);

% ==== Visualization ====
figure;
tiledlayout(5, 2, 'TileSpacing', 'compact');

% Row 1: Ridge Histogram
nexttile([1 2])
bar(all_vals_ridge, [count_ridge1, count_ridge2], 'grouped');
xlabel('Distance Values'); ylabel('Occurrences');
legend('Ridge 1', 'Ridge 2');
title('Histogram: Ridge Distances');

% Row 2: Bifurcation Histogram
nexttile([1 2])
bar(all_vals_bif, [count_bif1, count_bif2], 'grouped');
xlabel('Distance Values'); ylabel('Occurrences');
legend('Bif 1', 'Bif 2');
title('Histogram: Bifurcation Distances');

% Row 3: Combined Histogram
nexttile([1 2])
bar(all_vals_comb, [count_comb1, count_comb2], 'grouped');
xlabel('Distance Values'); ylabel('Occurrences');
legend('Combined 1', 'Combined 2');
title('Histogram: Ridge + Bifurcation Distances');

% Row 4: Fingerprint Images
nexttile
imshow(img1);
title('Fingerprint: Image 1');

nexttile
imshow(img2);
title('Fingerprint: Image 2');

% Summary Title
sgtitle(sprintf('Fingerprint Match: %.2f%% — %s', similarity * 100, state));


