% ------------------------------------------------------------------------------
% Fingerprint Recognition and Feature Extraction Project
% Copyright (c) 2025 Ofir Bar (ofirbar97@gmail.com)
%
% This code is licensed for personal and academic use only.
% Unauthorized reproduction or distribution of this code, or any portion of it,
% may result in severe civil and criminal penalties, and will be prosecuted
% to the maximum extent possible under law.
%
% Permission is granted to use, modify, and distribute this code for
% non-commercial and educational purposes provided that this notice
% is retained in all copies or substantial portions of the code.
%
% For commercial use, please contact: ofirbar97@gmail.com
% ------------------------------------------------------------------------------

%% Load the image
clc;
clear;
Image_name = 'raz1.bmp';

grayImage = imread(Image_name); %  image file

filename1 = sprintf('map_of_bif_%s.png',Image_name); % save file name as

filename2 = sprintf('map_of_ridge_%s.png',Image_name); % save file name as


figure;
% Original Image
subplot(1, 2, 2);
imshow(grayImage);
[h1, w1] = size(grayImage);
title(sprintf('Original Gray Image (%d × %d)', w1, h1));

% ROI Image
[roi_mask, img_roi] = extract_fingerprint_roi(grayImage);
subplot(1, 2,1);
imshow(img_roi);
[h2, w2] = size(img_roi);
title(sprintf('Fingerprint ROI (%d × %d)', w2, h2));
%%

grayImage=img_roi;

%%%[rows, cols] = size(grayImage);
%%%
%%%xCenter = round(cols / 2);  % x axis (columns)
%%%yCenter = round(rows / 2);  % y axis (rows)
%%%
%%%Zoom_windowSize = 125;  % amount to zoom in
%%%
%%%% Define bounds for cropping
%%%xStart = max(1, xCenter - Zoom_windowSize);
%%%xEnd   = min(cols, xCenter + Zoom_windowSize);
%%%yStart = max(1, yCenter - Zoom_windowSize);
%%%yEnd   = min(rows, yCenter + Zoom_windowSize);

% Crop the image around the center
%%%grayImage = grayImage(yStart:yEnd, xStart:xEnd);  % rows (y), cols (x)
imshow(grayImage);

%% Creating Average level and Substractin grayscaled image by neighborhood average level

% Initialize variables
[rows, cols] = size(grayImage); % Size of the grayscale image
After_low_I = zeros(rows, cols); % Preallocate for efficiency

% Define the neighborhood size  (windowSize is 'windowSize +1')
windowSize =5;

% Compute After_low_I by averaging the surrounding {windowSize x windowSize} window
for i = 1:rows
    for j = 1:cols
        % Determine the bounds of the window
        rStart = max(i - windowSize, 1);
        rEnd = min(i + windowSize, rows);
        cStart = max(j - windowSize, 1);
        cEnd = min(j + windowSize, cols);

        % Extract the neighborhood
        neighborhood = grayImage(rStart:rEnd, cStart:cEnd);

        % Compute the average of the neighborhood
        After_low_I(i, j) = mean(neighborhood(:));
    end
end

% Normalize the After_low_I to display properly
After_low_I = uint8(After_low_I);
subplot(1, 3, 1);
imshow(grayImage);
title('Original Gray Image');
% Display the smoothed image
subplot(1, 3, 2);
imshow(After_low_I);
title('Low Pass Image');

% Compute the difference image
differenceImage = grayImage- After_low_I;

% Display the difference image
subplot(1, 3, 3);
imshow(differenceImage, []);
title('Original Image - Low Pass Image');

%% Contrast stretching of the 'difference image' using a linear transformation 

minIntensity = double(min(differenceImage(:)));
maxIntensity = double(max(differenceImage(:)));

% Perform linear contrast stretching
adjustedImage = (double(differenceImage) - minIntensity) *(255 / (maxIntensity - minIntensity));

% Convert back to uint8
adjustedImage = uint8(adjustedImage);

figure;
%subplot(2, 1, 1);
%imshow(grayImage);
title('Original Gray Image');
% Display the smoothed image

% Display the difference image
subplot(1, 2, 1);
imshow(differenceImage, []);
title('Subtracted Image');

% Display the result
subplot(1,2, 2);
imshow(adjustedImage);
title('Contrast Adjusted using Linear Stretch');
%% Thresholding using Niblack method
windowSize = 6; % Local neighborhood size
k = 0.1;       % Niblack's parameter (typically negative)

% Convert image to double
img = double(adjustedImage);

% Create box filter for local mean
filterKernel = ones(windowSize, windowSize) / (windowSize^2);

% Compute local mean
meanImage = conv2(img, filterKernel, 'same');

% Compute local standard deviation
squaredImage = img.^2;
meanSquared = conv2(squaredImage, filterKernel, 'same');
stdImage = sqrt(meanSquared - meanImage.^2);

% Niblack threshold
threshold = meanImage + k * stdImage;

% Binarize image
after_binary = img > threshold;

% Optional post-processing
after_binary = ~after_binary;
after_binary = ~bwmorph(after_binary, 'fill', Inf);
figure;
% Display result
subplot(1,2, 1);
imshow(grayImage);
title('Original Gray Image');
subplot(1,2,2)
imshow(after_binary);
title('Niblack Binary Image');

%%
% removing white single pixels and 2x2 white cubic pixels
%--------------------------------------------------------
binaryImage = after_binary; % image 

% Define a 4x4 mask for detecting 2x2 white blocks
mask_4x4 = ones(4); % 4x4 block of 1s
mask =ones(3);
% looped 3 times to ensure removal of 2x2 and 1x1 white pixels
for i= 1:4
    % Apply convolution to compute neighborhood sums
    convResult = conv2(double(binaryImage), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
    
    % Output: Highlight or process single white pixels as needed
    % Remove single white pixels by setting them to 0 in the binary image
    binaryImage(singleWhitePixels) = 0;
    
    % Apply convolution to compute 4x4 block sums
    convResult_4x4 = conv2(double(binaryImage), mask_4x4, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 4 (indicating a 2x2 block of white pixels)
    twoByTwoWhitePixels = (convResult_4x4 <= 4)  & (binaryImage == 1);
    
    % Output: Highlight or process 4x4 white blocks as needed
    % Remove 2x2 white blocks by setting them to 0 in the binary image
    binaryImage(twoByTwoWhitePixels) = 0;
    
end  

% Display results
figure;
subplot(1, 2, 1);
imshow(binaryImage);
title('Removed Single White Pixels and 2x2 White Blocks');

%%
% removing black single pixels and 2x2 black cubic pixels
%--------------------------------------------------------

binaryImage= ~binaryImage; % inverting the image to use the same algo and invert back

% looped 3 times to ensure riddens of 2x2 and 1x1 white pixels
for i= 1:2
    % Apply convolution to compute neighborhood sums
    convResult = conv2(double(binaryImage), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
    
    % Output: Highlight or process single white pixels as needed
    % Remove single white pixels by setting them to 0 in the binary image
    binaryImage(singleWhitePixels) = 0;
    
    % Apply convolution to compute 4x4 block sums
    convResult_4x4 = conv2(double(binaryImage), mask_4x4, 'same'); 
    
    % Logical condition: Neighborhood sum is 4 (indicating a 2x2 block of white pixels)
    twoByTwoWhitePixels = (convResult_4x4 <= 4)  & (binaryImage == 1);
    
    % Output: Highlight or process 4x4 white blocks as needed
    % Remove 2x2 black blocks by setting them to 0 in the binary image
    binaryImage(twoByTwoWhitePixels) = 0;
    
end  
after_binary= ~binaryImage; % inverting the image to use the same algo and invert back


% Display results

subplot(1, 2,1);
imshow(after_binary);
title('Removed Single 1x1 and 2x2 pixels');
subplot(1, 2,2);
imshow(grayImage);
title('Original gray image');



%% prunnig edges process
%-------------------------------------------------------
% removing black 'edges'

after_binary = after_binary > 0; % making sure '0' or '1's
% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s
for j=1:3
% Initialize the result image
resultImage = after_binary;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(after_binary), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 5
% (i.e., the central pixel plus 4 surrounding white pixels)
singleWhitePixels = ((convResult == 4)) |(convResult == 5) & (after_binary == 0);

% Remove the single Black pixels by setting them to White 1
resultImage(singleWhitePixels) = 1;
end
% Apply convolution to compute neighborhood sums
convResult = conv2(double(resultImage), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 5
% (i.e., the central pixel plus 4 surrounding white pixels)
singleWhitePixels = ((convResult == 4)) |(convResult == 5) & (after_binary == 0);

% Remove the single Black pixels by setting them to White 1
resultImage(singleWhitePixels) = 1;
% Display the results
figure;

subplot(1, 2, 1);
imshow(resultImage);
title('Pruned image');

subplot(1, 2, 2);
imshow(after_binary);
title('Pre prunned Binary image');
pruned_image = resultImage;
%%%%%%%
%%%%%%-------------------------------------------------------
%%%%%% removing white 'edges'
%%%%%
%%%%%% Assuming binaryImage is the binary image (1 for white, 0 for black)
%%%%%binaryImage = ~resultImage; % Example image
%%%%%
%%%%%% Define a 3x3 mask for detecting the neighborhood of each pixel
%%%%%mask = ones(3); % 3x3 block of 1s
%%%%%
%%%%%% Apply convolution to compute neighborhood sums
%%%%%convResult = conv2(double(binaryImage), mask, 'same'); % Convolve the image with the mask
%%%%%
%%%%%% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 4
%%%%%% (i.e., the central pixel plus 3 surrounding white pixels)
%%%%%singleBlackPixels  = (convResult == 4) & (after_binary == 1);
%%%%%
%%%%%% Remove the single white pixels by setting them to black (0)
%%%%%resultImage(singleBlackPixels) = 0;
%%%%%%resultImage = ~resultImage; % using 'NOT' to change back the colors
%%%%%
%%%%%% Display the results
%%%%%subplot(1, 2, 2);
%%%%%imshow(resultImage);
%%%%%title('Removed white  (branches)');

%% pre thinning
pre_thinned = remove_noise_blobs(pruned_image);
imshow(pre_thinned);
%% post pre pre 
cleaned = remove_black_edges(pre_thinned);    % Apply the function
imshow(cleaned);  
title('Pre-thinned Image')% Display result
%% Thinning Process


%thin_image=bwmorph(~binaryImage,'clean',inf);
thin_image=~bwmorph(~resultImage,'thin',inf);
thin_image=~bwmorph(thin_image,'spur',inf);

thin_image=~thin_image;
figure;
%subplot(1, 2, 1);
%imshow(grayImage);
title('Original Gray Image');
subplot(1, 1, 1);
imshow(thin_image);
title('Thinned Image');



%% Removing Branches Created by the Thinning process
%--------------------------------------------------------

thin_image_2= ~thin_image; % inverting the image to use the same algo and invert back

% discards 2x2 and 1x1 white pixels
for i= 1:5
    % Apply convolution to compute neighborhood sums
    convResult_2 = conv2(double(thin_image_2), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult_2 == 2) & (thin_image_2 == 1);
    
    % Output: Highlight or process single white pixels as needed
    % Remove single white pixels by setting them to 0 in the binary image
    thin_image_2(singleWhitePixels) = 0;
    
    % Apply convolution to compute 4x4 block sums
    convResult_4x4 = conv2(double(thin_image_2), mask_4x4, 'same'); 
    
    % Logical condition: Neighborhood sum is 4 (indicating a 2x2 block of white pixels)
    twoByTwoWhitePixels = (convResult_4x4 <= 2)  & (thin_image_2 == 1);
    
    % Output: Highlight or process 4x4 white blocks as needed
    % Remove 2x2 black blocks by setting them to 0 in the binary image
    binaryImage(twoByTwoWhitePixels) = 0;
    
end  


thin_image_2 = ~thin_image_2;



% Display results
figure
imshow(thin_image_2);
title('Thinned Image');
thin_image_2 =~thin_image_2; % for applying functions on '1's and not '0's
% since its impossible to works with sum when ridges are '0'.


%% post thinning 
% Removing  REISH ,ר, figures
post_thin = removePatternedPixels(thin_image_2);
%post_thin=~post_thin;
figure;
imshow(~post_thin);
title('Thinned image spurs removed')

%%
[roi_mask2, cleaned_image] = extract_fingerprint_roi(double(post_thin));
figure;
imshow(cleaned_image);
[h2, w2] = size(roi_mask2);
title(sprintf('Cleaned Fingerprint (%d × %d)', w2, h2));
%% detecting bifurcations
bifurcations = detectBifurcations(cleaned_image);
figure;

% Get coordinates of bifurcations
[y_bif, x_bif] = find(bifurcations);  % y = row, x = column
map_bif = [x_bif, y_bif];
% Display the thinned image

imshow(~cleaned_image);
hold on;

% Overlay red dots on bifurcation locations
plot(x_bif, y_bif, 'r.', 'MarkerSize', 10); 
title('Detected Bifurcation Points on Thinned Image');

%% Detecting ridges
% Call the ridge detection function
[ridgeMap,ridgeCoords] = detectRidgeSegments(cleaned_image);
[y_rid, x_rid] = find(ridgeMap);
map_ridge = [y_rid, x_rid];
% Display the thinned image
figure;
imshow(~cleaned_image);
title('Detected Ridge Segment Points');
hold on;

% Plot the ridge coordinates on top of the image
plot(ridgeCoords(:,2), ridgeCoords(:,1), 'b.', 'MarkerSize', 14); % X = column, Y = row

hold off;

%%

%% Detecting Ridges
% Call the ridge detection function
[ridgeMap, ridgeCoords] = detectRidgeSegments(cleaned_image);
% Display the thinned image
figure;
%subplot(1,2,1)
imshow(~cleaned_image);
hold on;
% Overlay red dots on bifurcation locations
plot(x_bif, y_bif, 'r.', 'MarkerSize', 10);
% Overlay blue dots on ridge segment locations
plot(ridgeCoords(:,2), ridgeCoords(:,1), 'b.', 'MarkerSize', 14); % X = column, Y = row
title('Detected Bifurcation and Ridge Segment Points');
legend('Bifurcations','Ridges');
hold off;
figure;

imshow(grayImage)
hold on;
% Overlay red dots on bifurcation locationsb 
plot(x_bif, y_bif, 'r.', 'MarkerSize', 12);
% Overlay blue dots on ridge segment locations
plot(ridgeCoords(:,2), ridgeCoords(:,1), 'b.', 'MarkerSize', 16); % X = column, Y = row
title('Original Image with Detected Features');
legend('Bifurcations','Ridges');
hold off;
%% Exporting to Map of ones of ridges and bifurcations
map_size = size(grayImage);
map_of_ones_bif = create_map_from_coordinates(map_bif, map_size);
map_of_ones_ridges = create_map_from_coordinates(map_ridge, map_size);
%% Optional Export to CSV File
%%%% Exporting to CSV file with [X coordinates, Y coordinates,Minutiae type (1 OR 3) ,dist_ridges, dist_bifurcation, dist_combind]
%%%ist_ridge = shortest_paths_to_nearest_one_diag(map_of_ones_ridges);
%%%ist_bif   = shortest_paths_to_nearest_one_diag(map_of_ones_bif);
%%%ist_combined =shortest_paths_to_nearest_one_diag(map_of_ones_ridges + map_of_ones_bif);
%%%if_coor = [x_bif, y_bif, 3 * ones(length(x_bif), 1)];
%%%idge_coor = [x_rid, y_rid, ones(length(x_rid), 1)];
%%%oordinates = [bif_coor; ridge_coor];
%%% Assuming dist_ridge, dist_bif, dist_combined are the input matrices
%%%ax_len = max([length(dist_ridge), length(dist_bif), length(dist_combined),length(coordinates)]);
%%%
%%% Pad each matrix with NaNs to the same length
%%%ist_ridge_padded = pad_vector(dist_ridge, max_len);
%%%ist_bif_padded = pad_vector(dist_bif, max_len);
%%%ist_combined_padded = pad_vector(dist_combined, max_len);
%%%
%%% Now concatenate them into one matrix
%%%istances = [dist_ridge_padded, dist_bif_padded, dist_combined_padded];
%%%
%%%inal_data = [coordinates, distances];
%%%olumn_names = {'X_Coordinate', 'Y_Coordinate', 'Type', 'Dist_Ridge', 'Dist_Bif', 'Dist_Combined'};
%%%
%%% Convert final_data to a table
%%%inal_table = array2table(final_data, 'VariableNames', column_names);
%%% Export the result to CSV
%%%writematrix(final_data, 'minutiae_data_ofir_7_2.csv');

%% write map of ones as png

%imwrite(map_of_ones_bif, filename1);

%imwrite(map_of_ones_ridges, filename2);
%% Display Distances from nearest minutiae point as histogram of distances over occurences.

% Enhanced Fingerprint Distance Analysis
fingerprint_id = Image_name; % Replace with actual fingerprint ID

% Calculate distances
dist_ridge = shortest_paths_to_nearest_one_diag(map_of_ones_ridges);
dist_bif = shortest_paths_to_nearest_one_diag(map_of_ones_bif);
combined = [dist_ridge(:); dist_bif(:)];

% Figure 1: Combined Ridge + Bifurcation Histogram
all_vals_comb = unique(combined);
count_comb1 = histc(combined, all_vals_comb);

figure('Position', [100, 100, 800, 600]);
b = bar(all_vals_comb, count_comb1, 'grouped', 'FaceAlpha', 0.8, 'EdgeColor', 'black', 'LineWidth', 1.2);
b.FaceColor = [0.2 0.6 0.8]; % Nice blue color

% Enhanced styling
xlabel('Distance Values', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Occurrences', 'FontSize', 12, 'FontWeight', 'bold');

% Multi-line title with fingerprint ID
title({['Fingerprint ID: ' fingerprint_id], ...
       'Distance Distribution: Ridge + Bifurcation Features'}, ...
       'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.5]);

% Add grid and improve appearance
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);
set(gca, 'FontSize', 10, 'LineWidth', 1);

% Add statistics text box
mean_dist = mean(combined);
std_dist = std(combined);
max_count = max(count_comb1);
total_points = length(combined);

stats_text = sprintf('Statistics:\nMean: %.2f\nStd: %.2f\nMax Count: %d\nTotal Points: %d', ...
                    mean_dist, std_dist, max_count, total_points);

% Position text box in upper right
text(0.75, 0.9, stats_text, 'Units', 'normalized', ...
     'BackgroundColor', [1 1 0.9], 'EdgeColor', 'black', ...
     'FontSize', 9, 'VerticalAlignment', 'top', ...
     'HorizontalAlignment', 'left');

% Enhance axes limits for better visualization
xlim([min(all_vals_comb)-0.5, max(all_vals_comb)+0.5]);
ylim([0, max(count_comb1)*1.1]);

% Figure 2: Ridge-Only Histogram
all_vals_ridge = unique(dist_ridge(:));
count_ridge = histc(dist_ridge(:), all_vals_ridge);

figure('Position', [200, 200, 800, 600]);
b_ridge = bar(all_vals_ridge, count_ridge, 'grouped', 'FaceAlpha', 0.8, 'EdgeColor', 'black', 'LineWidth', 1.2);
b_ridge.FaceColor = [0.8 0.2 0.2]; % Red color for ridges

% Enhanced styling
xlabel('Distance Values', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Occurrences', 'FontSize', 12, 'FontWeight', 'bold');

% Multi-line title with fingerprint ID
title({['Fingerprint ID: ' fingerprint_id], ...
       'Distance Distribution: Ridge Features Only'}, ...
       'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.5 0.1 0.1]);

% Add grid and improve appearance
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);
set(gca, 'FontSize', 10, 'LineWidth', 1);

% Add statistics text box for ridges
mean_ridge = mean(dist_ridge(:));
std_ridge = std(dist_ridge(:));
max_count_ridge = max(count_ridge);
total_ridge_points = length(dist_ridge(:));

stats_text_ridge = sprintf('Ridge Statistics:\nMean: %.2f\nStd: %.2f\nMax Count: %d\nTotal Points: %d', ...
                          mean_ridge, std_ridge, max_count_ridge, total_ridge_points);

% Position text box in upper right
text(0.75, 0.9, stats_text_ridge, 'Units', 'normalized', ...
     'BackgroundColor', [1 0.9 0.9], 'EdgeColor', 'red', ...
     'FontSize', 9, 'VerticalAlignment', 'top', ...
     'HorizontalAlignment', 'left');

% Enhance axes limits for better visualization
xlim([min(all_vals_ridge)-0.5, max(all_vals_ridge)+0.5]);
ylim([0, max(count_ridge)*1.1]);

% Figure 3: Bifurcation-Only Histogram
all_vals_bif = unique(dist_bif(:));
count_bif = histc(dist_bif(:), all_vals_bif);

figure('Position', [300, 300, 800, 600]);
b_bif = bar(all_vals_bif, count_bif, 'grouped', 'FaceAlpha', 0.8, 'EdgeColor', 'black', 'LineWidth', 1.2);
b_bif.FaceColor = [0.2 0.8 0.2]; % Green color for bifurcations

% Enhanced styling
xlabel('Distance Values', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Occurrences', 'FontSize', 12, 'FontWeight', 'bold');

% Multi-line title with fingerprint ID
title({['Fingerprint ID: ' fingerprint_id], ...
       'Distance Distribution: Bifurcation Features Only'}, ...
       'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.1 0.5 0.1]);

% Add grid and improve appearance
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);
set(gca, 'FontSize', 10, 'LineWidth', 1);

% Add statistics text box for bifurcations
mean_bif = mean(dist_bif(:));
std_bif = std(dist_bif(:));
max_count_bif = max(count_bif);
total_bif_points = length(dist_bif(:));

stats_text_bif = sprintf('Bifurcation Statistics:\nMean: %.2f\nStd: %.2f\nMax Count: %d\nTotal Points: %d', ...
                        mean_bif, std_bif, max_count_bif, total_bif_points);

% Position text box in upper right
text(0.75, 0.9, stats_text_bif, 'Units', 'normalized', ...
     'BackgroundColor', [0.9 1 0.9], 'EdgeColor', 'green', ...
     'FontSize', 9, 'VerticalAlignment', 'top', ...
     'HorizontalAlignment', 'left');

% Enhance axes limits for better visualization
xlim([min(all_vals_bif)-0.5, max(all_vals_bif)+0.5]);
ylim([0, max(count_bif)*1.1]);

% Display summary information
fprintf('\n=== FINGERPRINT ANALYSIS SUMMARY ===\n');
fprintf('Fingerprint ID: %s\n', fingerprint_id);
fprintf('Combined Features - Mean: %.2f, Std: %.2f, Total Points: %d\n', mean_dist, std_dist, total_points);
fprintf('Ridge Features - Mean: %.2f, Std: %.2f, Total Points: %d\n', mean_ridge, std_ridge, total_ridge_points);
fprintf('Bifurcation Features - Mean: %.2f, Std: %.2f, Total Points: %d\n', mean_bif, std_bif, total_bif_points);
fprintf('=====================================\n');
