clear; clc;

% Choose similarity metric
similarity_method = 'bhattacharyya'; % Options: intersection, chi_square, bhattacharyya, cosine

% ==== Read and process fingerprint 4 (reference) ====
ridge4 = double(imread('map_of_ridge_ofir_3_4.png') > 0);
bif4 = double(imread('map_of_bif_ofir_3_4.png') > 0);
img4 = imread('ofir_7_1.bmp');
dist_ridge4 = shortest_paths_to_nearest_one_diag(ridge4);
dist_bif4 = shortest_paths_to_nearest_one_diag(bif4);

% Store comparison results
comparisons = [];
fingerprint_names = {'1', '2', '3'};

% Compare each fingerprint (1, 2, 3) with fingerprint 4
for i = 1:3
    % Read fingerprint i
    ridge_i = double(imread(sprintf('map_of_ridge_ofir_7_%d.png', i)) > 0);
    bif_i = double(imread(sprintf('map_of_bif_ofir_7_%d.png', i)) > 0);
    img_i = imread(sprintf('index_L%d.bmp', i));
    dist_ridge_i = shortest_paths_to_nearest_one_diag(ridge_i);
    dist_bif_i = shortest_paths_to_nearest_one_diag(bif_i);
    
    % Compare fingerprint i with fingerprint 4
    [similarity, match_ridge, match_comb, w_ridge, w_comb] = compare_fingerprints(...
        ridge_i, bif_i, ridge4, bif4, dist_ridge_i, dist_bif_i, dist_ridge4, dist_bif4, similarity_method);
    
    % Store results
    comparisons(i).fingerprint = fingerprint_names{i};
    comparisons(i).similarity = similarity;
    comparisons(i).match_ridge = match_ridge;
    comparisons(i).match_comb = match_comb;
    comparisons(i).w_ridge = w_ridge;
    comparisons(i).w_comb = w_comb;
    comparisons(i).img = img_i;
    
    % Display individual results
    fprintf('Fingerprint %s vs 4:\n', fingerprint_names{i});
    fprintf('  Ridge Similarity: %.4f (Weight: %.2f)\n', match_ridge, w_ridge);
    fprintf('  Combined Similarity: %.4f (Weight: %.2f)\n', match_comb, w_comb);
    fprintf('  Overall Similarity: %.4f\n\n', similarity);
end

% Set fixed confidence thresholds
similarity_scores = [comparisons.similarity];
low_threshold = 0.66;   % 70% for low confidence
high_threshold = 0.70;  % 80% for high confidence

% Display threshold and validation results
fprintf('=== CONFIDENCE THRESHOLDS ===\n');
fprintf('Low Confidence Threshold: %.0f%%\n', low_threshold * 100);
fprintf('High Confidence Threshold: %.0f%%\n', high_threshold * 100);
fprintf('Individual Scores: [%.4f, %.4f, %.4f]\n', similarity_scores);

% Validate each comparison against confidence thresholds
fprintf('\n=== VALIDATION RESULTS ===\n');
votes = zeros(1, 3);  % Store vote weights for each fingerprint
vote_types = cell(1, 3);  % Store vote types for display

for i = 1:3
    score_percent = comparisons(i).similarity * 100;
    if comparisons(i).similarity >= high_threshold
        validation_result = 'HIGH CONFIDENCE MATCH';
        confidence_level = 'High';
        votes(i) = 2.0;  % High confidence vote weight
        vote_types{i} = 'High Conf Match';
    elseif comparisons(i).similarity >= low_threshold
        validation_result = 'LOW CONFIDENCE MATCH';
        confidence_level = 'Low';
        votes(i) = 1.0;  % Low confidence vote weight
        vote_types{i} = 'Low Conf Match';
    else
        validation_result = 'NO MATCH';
        confidence_level = 'None';
        votes(i) = 0.0;  % No match vote weight
        vote_types{i} = 'No Match';
    end
    fprintf('Fingerprint %s vs 4: %.1f%% - %s (%s Confidence) [Vote Weight: %.1f]\n', ...
        fingerprint_names{i}, score_percent, validation_result, confidence_level, votes(i));
end

% ==== WEIGHTED MAJORITY VOTING ====
fprintf('\n=== WEIGHTED MAJORITY VOTING ===\n');

% Calculate voting statistics
total_votes = sum(votes);
max_possible_votes = 3 * 2.0;  % 3 fingerprints × 2.0 max weight each
high_conf_votes = sum(votes == 2.0);
low_conf_votes = sum(votes == 1.0);
no_match_votes = sum(votes == 0.0);
vote_percentage = (total_votes / max_possible_votes) * 100;

% Majority voting thresholds
majority_threshold = 0.50;  % 50% of total possible votes
strong_majority_threshold = 0.67;  % 67% of total possible votes (2/3)

fprintf('Vote Distribution:\n');
fprintf('  High Confidence Votes (2.0): %d\n', high_conf_votes);
fprintf('  Low Confidence Votes (1.0):  %d\n', low_conf_votes);
fprintf('  No Match Votes (0.0):        %d\n', no_match_votes);
fprintf('  Total Weighted Votes: %.1f / %.1f (%.1f%%)\n', total_votes, max_possible_votes, vote_percentage);

% Determine majority voting result
if vote_percentage >= strong_majority_threshold * 100
    majority_decision = 'STRONG MATCH';
    decision_confidence = 'Very High';
    decision_color = 'green';
elseif vote_percentage >= majority_threshold * 100
    majority_decision = 'WEAK MATCH';
    decision_confidence = 'Moderate';
    decision_color = 'orange';
else
    majority_decision = 'NO MATCH';
    decision_confidence = 'Low';
    decision_color = 'red';
end

fprintf('\nMajority Voting Decision:\n');
fprintf('  Majority Threshold: %.0f%% | Strong Majority Threshold: %.0f%%\n', ...
    majority_threshold * 100, strong_majority_threshold * 100);
fprintf('  FINAL DECISION: %s (Confidence: %s)\n', majority_decision, decision_confidence);
fprintf('  Vote Percentage: %.1f%%\n', vote_percentage);

% Additional consensus analysis
consensus_score = std(similarity_scores);  % Lower std = higher consensus
fprintf('  Consensus Score: %.4f (lower = more agreement)\n', consensus_score);

% ==== Visualization ====
figure;
tiledlayout(4, 4, 'TileSpacing', 'compact');

% Row 1: Show fingerprint 4 (reference)
nexttile([1 4])
imshow(img4);
title('Reference Fingerprint 4');

% Rows 2-4: Show comparisons
for i = 1:3
    % Fingerprint image
    nexttile
    imshow(comparisons(i).img);
    title(sprintf('Fingerprint %s', fingerprint_names{i}));
    
    % Similarity score bar
    nexttile
    bar([comparisons(i).match_ridge, comparisons(i).match_comb], 'grouped');
    set(gca, 'XTickLabel', {'Ridge', 'Combined'});
    ylabel('Similarity Score');
    title(sprintf('Scores: Ridge=%.3f, Comb=%.3f', ...
        comparisons(i).match_ridge, comparisons(i).match_comb));
    ylim([0 1]);
    
    % Overall similarity with threshold lines
    nexttile
    bar(comparisons(i).similarity, 'FaceColor', [0.2 0.6 0.8]);
    hold on;
    plot([0.5 1.5], [low_threshold low_threshold], 'y--', 'LineWidth', 2);
    plot([0.5 1.5], [high_threshold high_threshold], 'r--', 'LineWidth', 2);
    ylabel('Overall Similarity');
    title(sprintf('Overall: %.1f%% (%s)', ...
        comparisons(i).similarity * 100, vote_types{i}));
    xlim([0.5 1.5]);
    ylim([0 1]);
    legend('Similarity', 'Low Threshold (70%)', 'High Threshold (80%)', 'Location', 'best');
    
    % Vote weight visualization
    nexttile
    if votes(i) == 0
        bar_color = 'red';
    elseif votes(i) == 1
        bar_color = 'yellow';
    else  % votes(i) == 2
        bar_color = 'green';
    end
    bar(votes(i), 'FaceColor', bar_color);
    ylabel('Vote Weight');
    title(sprintf('Vote: %.1f', votes(i)));
    ylim([0 2.5]);
    hold on;
    plot([0.5 1.5], [1.0 1.0], 'y--', 'LineWidth', 1);
    plot([0.5 1.5], [2.0 2.0], 'g--', 'LineWidth', 1);
    xlim([0.5 1.5]);
end

% Summary title with majority decision
sgtitle(sprintf('Fingerprint Validation - MAJORITY DECISION: %s (%.1f%% votes)', ...
    majority_decision, vote_percentage));

% ==== Create Majority Voting Summary Plot ====
figure;
tiledlayout(2, 2, 'TileSpacing', 'compact');

% Pie chart of vote distribution
nexttile
vote_counts = [high_conf_votes, low_conf_votes, no_match_votes];
vote_labels = {'High Conf (2.0)', 'Low Conf (1.0)', 'No Match (0.0)'};
pie(vote_counts, vote_labels);
title('Vote Distribution by Type');

% Bar chart of weighted votes
nexttile
bar(1:3, votes, 'FaceColor', [0.3 0.7 0.9]);
hold on;
plot([0.5 3.5], [1.0 1.0], 'y--', 'LineWidth', 2);
plot([0.5 3.5], [2.0 2.0], 'g--', 'LineWidth', 2);
xlabel('Fingerprint');
ylabel('Vote Weight');
title('Individual Vote Weights');
set(gca, 'XTickLabel', fingerprint_names);
ylim([0 2.5]);
legend('Vote Weight', 'Low Threshold', 'High Threshold', 'Location', 'best');

% Total vote percentage gauge
nexttile
theta = linspace(0, pi, 100);
x_outer = cos(theta);
y_outer = sin(theta);
x_inner = 0.7 * cos(theta);
y_inner = 0.7 * sin(theta);

fill([x_outer, fliplr(x_inner)], [y_outer, fliplr(y_inner)], [0.9 0.9 0.9]);
hold on;

% Color code the gauge based on decision
gauge_color = [0.8 0.2 0.2];  % red RGB
if strcmp(majority_decision, 'STRONG MATCH')
    gauge_color = [0.2 0.8 0.2];  % green RGB
elseif strcmp(majority_decision, 'WEAK MATCH')
    gauge_color = [1.0 0.6 0.0];  % orange RGB
end

vote_angle = pi * (1 - vote_percentage/100);
plot([0, cos(vote_angle)], [0, sin(vote_angle)], 'Color', gauge_color, 'LineWidth', 6);
text(0, -0.3, sprintf('%.1f%%', vote_percentage), 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
title('Total Vote Percentage');
axis equal;
axis([-1.2 1.2 -0.5 1.2]);
axis off;

% Decision summary text
nexttile
axis off;
decision_text_color = [0.8 0.2 0.2];  % red RGB
if strcmp(majority_decision, 'STRONG MATCH')
    decision_text_color = [0.2 0.8 0.2];  % green RGB
elseif strcmp(majority_decision, 'WEAK MATCH')
    decision_text_color = [1.0 0.6 0.0];  % orange RGB
end

text(0.1, 0.8, 'MAJORITY VOTING RESULT:', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.6, sprintf('Decision: %s', majority_decision), 'FontSize', 12, 'FontWeight', 'bold', 'Color', decision_text_color);
text(0.1, 0.4, sprintf('Confidence: %s', decision_confidence), 'FontSize', 12);
text(0.1, 0.2, sprintf('Vote Percentage: %.1f%%', vote_percentage), 'FontSize', 12);
text(0.1, 0.0, sprintf('Consensus Score: %.4f', consensus_score), 'FontSize', 12);

sgtitle('Weighted Majority Voting Summary');

% ==== Summary Statistics ====
fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('Similarity Method: %s\n', similarity_method);
fprintf('Individual Thresholds - Low: %.0f%%, High: %.0f%%\n', low_threshold * 100, high_threshold * 100);
fprintf('Majority Thresholds - Simple: %.0f%%, Strong: %.0f%%\n', majority_threshold * 100, strong_majority_threshold * 100);
fprintf('Standard Deviation: %.4f\n', std(similarity_scores));
fprintf('Range: [%.1f%%, %.1f%%]\n', min(similarity_scores) * 100, max(similarity_scores) * 100);
fprintf('Consensus Score: %.4f (lower = higher agreement)\n', consensus_score);

% Count results by confidence level
fprintf('\nDetailed Vote Analysis:\n');
fprintf('High Confidence Matches: %d/%d (%.1f votes)\n', high_conf_votes, length(similarity_scores), high_conf_votes * 2.0);
fprintf('Low Confidence Matches: %d/%d (%.1f votes)\n', low_conf_votes, length(similarity_scores), low_conf_votes * 1.0);
fprintf('No Matches: %d/%d (%.1f votes)\n', no_match_votes, length(similarity_scores), no_match_votes * 0.0);
fprintf('Total Weighted Votes: %.1f / %.1f\n', total_votes, max_possible_votes);

fprintf('\n=== FINAL RECOMMENDATION ===\n');
fprintf('Based on weighted majority voting across %d fingerprints:\n', length(fingerprint_names));
fprintf('DECISION: %s\n', majority_decision);
fprintf('CONFIDENCE: %s\n', decision_confidence);
fprintf('REASONING: %.1f%% of possible votes achieved\n', vote_percentage);