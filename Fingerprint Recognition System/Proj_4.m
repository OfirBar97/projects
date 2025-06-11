%% Load the image
clc;
clear;

I = imread('ofir_3.bmp'); % Replace with your image file
filename = sprintf('map_of_ones_ofir_2.png'); % write map of ones at the end
max_pixel = max(max(max(I)));
if max_pixel>255
    grayImage = rgb2gray(I); % Convert to grayscale
else 
    grayImage = im2gray(I);
end

[rows, cols] = size(grayImage);

xCenter = round(cols / 2);  % x axis (columns)
yCenter = round(rows / 2);  % y axis (rows)

Zoom_windowSize = 100;  % amount to zoom in

% Define bounds for cropping
xStart = max(1, xCenter - Zoom_windowSize);
xEnd   = min(cols, xCenter + Zoom_windowSize);
yStart = max(1, yCenter - Zoom_windowSize);
yEnd   = min(rows, yCenter + Zoom_windowSize);

% Crop the image around the center
grayImage = grayImage(yStart:yEnd, xStart:xEnd);  % rows (y), cols (x)
figure;
subplot(3, 2, 1);
imshow(grayImage);
title('Original Gray Image');

%xlim([190.480105229998 379.74681094995]);
% Uncomment the following line to preserve the Y-limits of the axes
%ylim([253.062312374234 489.996188423656]);
%grayImage=grayImage(250:858,150:582);
%grayImage=grayImage(253.062312374234 : 489.996188423656,190.480105229998: 379.74681094995);


%%
%enhancedImage = fftFingerprintEnhancer(grayImage);
%grayImage=enhancedImage;
%% Creating Average level and Substractin grayscaled image by neighborhood average level

% Initialize variables
[rows, cols] = size(grayImage); % Size of the grayscale image
After_low_I = zeros(rows, cols); % Preallocate for efficiency

% Define the neighborhood size  (windowSize is 'windowSize +1')
windowSize = 10;

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

% Display the smoothed image
subplot(3, 2, 2);
imshow(After_low_I);
title('After high filter I');

% Compute the difference image
differenceImage = grayImage- After_low_I;

% Display the difference image
subplot(3, 2, 3);
imshow(differenceImage, []);
title('After grayImage - After High I');
%%
%enhancedImage = fftFingerprintEnhancer(differenceImage);
%differenceImage=enhancedImage;
%% Contrast stretching of the 'difference image' using a linear transformation 

minIntensity = double(min(differenceImage(:)));
maxIntensity = double(max(differenceImage(:)));

% Perform linear contrast stretching
adjustedImage = (double(differenceImage) - minIntensity) *2*(255 / (maxIntensity - minIntensity));

% Convert back to uint8
adjustedImage = uint8(adjustedImage);

% Display the result
subplot(3, 2, 4);
imshow(adjustedImage);
title('Contrast Adjusted using Linear Stretch');

%% Thresholding function -> Binary image

% Define the neighborhood size (i.e., window size)
windowSize = 10; % Adjust as needed
halfWindow = floor(windowSize / 2);

% Create an averaging filter (box filter) for neighborhood mean calculation
filterKernel = ones(windowSize, windowSize) / (windowSize^2);

% Compute mean of the surrounding window using convolution
meanImage = conv2(double(adjustedImage), filterKernel, 'same');

% Perform thresholding
after_binary = adjustedImage <= meanImage;
after_binary =~after_binary;

% Display the binary image
subplot(3, 2, 5);
imshow(after_binary);

title('Binary Image');



%%
% removing white single pixels and 2x2 white cubic pixels
%--------------------------------------------------------
mask = ones(3);
binaryImage = after_binary; % image 

% Define a 4x4 mask for detecting 2x2 white blocks
mask_4x4 = ones(4); % 4x4 block of 1s

% looped 3 times to ensure removal of 2x2 and 1x1 white pixels
for i= 1:3

    binaryImage= ~binaryImage;
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
binaryImage= ~binaryImage;

% Display results
figure;
subplot(1, 2, 1);
imshow(binaryImage);
title('Removed Single White Pixels and 2x2 White Blocks');

%%

% removing black single pixels and 2x2 black cubic pixels
%--------------------------------------------------------

 % inverting the image to use the same algo and invert back

% looped 3 times to ensure riddens of 2x2 and 1x1 white pixels
for i= 1:6
    % Apply convolution to compute neighborhood sums
    binaryImage= ~binaryImage;

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

%binaryImage = ~binaryImage;
% Display results

subplot(1, 2,2);
imshow(binaryImage);
title('Removed Single Black Pixels and 2x2 Black Blocks');
%% prunnig edges process
%-------------------------------------------------------
% removing White 'edges'

after_binary = binaryImage > 0; % making sure '0' or '1's
% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s

% Initialize the result image
resultImage = after_binary;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(after_binary), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 5
% (i.e., the central pixel plus 4 surrounding white pixels)
singleWhiteEdges = (convResult == 4) & (after_binary == 1);

% Remove the single wihte pixels by setting them to black 0
resultImage(singleWhiteEdges) = 0;

% Display the results
figure;

subplot(1, 2, 1);
imshow(resultImage);
title('Removed black  (branches)');

%%
%-------------------------------------------------------
% removing Black 'edges'
resultImage = ~resultImage; % using 'NOT' to change back the colors

% Assuming binaryImage is the binary image (1 for white, 0 for black)
binaryImage = resultImage; % Example image

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 4
% (i.e., the central pixel plus 3 surrounding white pixels)
singleBlackEdges  = (convResult == 4) & (after_binary == 1);

% Remove the single black pixels by setting them to white (1)
resultImage(singleBlackEdges) = 1;
pre_thinning = ~resultImage; % using 'NOT' to change back the colors

% Display the results
subplot(1, 2, 2);
imshow(pre_thinning);
title('Removed white  (branches)');


%%

 % inverting the image to use the same algo and invert back

% discard 2x2 and 1x1 black and white pixels
for i= 1:4
    % Apply convolution to compute neighborhood sums
    pre_thinning= ~pre_thinning;

    convResult = conv2(double(pre_thinning), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult == 1) & (pre_thinning == 1);
    
    % Output: Highlight or process single white pixels as needed
    % Remove single white pixels by setting them to 0 in the binary image
    pre_thinning(singleWhitePixels) = 0;
    
    % Apply convolution to compute 4x4 block sums
    convResult_4x4 = conv2(double(pre_thinning), mask_4x4, 'same'); 
    
    % Logical condition: Neighborhood sum is 4 (indicating a 2x2 block of white pixels)
    twoByTwoWhitePixels = (convResult_4x4 <= 4)  & (pre_thinning == 1);
    
    % Output: Highlight or process 4x4 white blocks as needed
    % Remove 2x2 black blocks by setting them to 0 in the binary image
    pre_thinning(twoByTwoWhitePixels) = 0;
    
end  

%binaryImage = ~binaryImage;
% Display results
figure;
imshow(pre_thinning);

%% Thinning Process


%
figure;
subplot(1, 2, 1);
%thin_image=bwmorph(~binaryImage,'clean',inf);
thin_image=~bwmorph(~pre_thinning,'thin',inf);
thin_image=~bwmorph(thin_image,'spur',inf);

thin_image=~thin_image;
imshow(thin_image);
title('Thinned Image');


subplot(1, 2, 2);
imshow(grayImage);
title('Original Gray Image');
%% Removing Branches Created by the Thinning process
%--------------------------------------------------------

thin_image_2= ~thin_image; % inverting the image to use the same algo and invert back

% discards 2x2 and 1x1 white pixels
for i= 1:5
    % Apply convolution to compute neighborhood sums
    convResult = conv2(double(thin_image_2), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult == 1) & (thin_image_2 == 1);
    
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
subplot(1,2,1)
imshow(thin_image_2);
title('Thinned Image v2');
subplot(1,2,2)
imshow(grayImage);

%thin_image_3=bwmorph(thin_image_2,'spur',inf);
%imshow(thin_image_3);

%% Idea
% trying to extract useful information, using extraction of x,y
% coordiantes, then tr
% ansforming into a 2D list, and number of points.
% Matrix  A = first_register of image;
% Matrix  B = second_image for validation;
% C = number of A[x,y] that exists in B[x,y]
% D = number of B[x,y] that exists in A[x,y]
% NB = number of elements in  B, NA = number of elements in  A
% calculation -> C+D/(A+B) *100 = percentage of occurences matching
% Initialize lists for ridge endings (A) and bifurcations (B)

%% creating new image 'result' that contains ridges end and bifurcation, pre-filtered.

% Get image size
[rows, cols] = size(thin_image_2);

% Initialize result matrix with zeros
result = zeros(rows, cols);

% Loop through the image, avoiding the borders
for i = 10:rows-10
    for j = 10:cols-10
        if thin_image_2(i, j) == 0  % Check only ridge points
            % Extract 3x3 neighborhood
            neighborhood = thin_image_2(i-1:i+1, j-1:j+1);
            
            % Count black pixels (zeros)
            black_pixel_count = sum(neighborhood(:));
            
            % Apply the conditions
            if black_pixel_count == 7
                result(i, j) = 1;  % Ridge ending
            end
                
                
            if black_pixel_count ==  5
                result(i, j) = 1;  % Bifurcation
            end
            
        end
    end
end
%% filtering neighboring intreset points to singles

[rows, cols] = size(result);
filtered_result = result; % Copy the original image

% Loop through the image with a 3x3 sliding window
for i = 2:rows-1
    for j = 2:cols-1
        % Extract the 3x3 window
        window = result(i-1:i+1, j-1:j+1);
        
        % Check if there are exactly 3 ones in the window
        if sum(window(:)) == 3
            % Reduce to a single '1' (choose priority: center, top-left, or random)
            newWindow = zeros(3,3); % Reset the 3x3 window to all zeros
            
            if window(2,2) == 1
                newWindow(2,2) = 1; % Keep center pixel if it exists
            else
                indices = find(window == 1); % Find indices of '1' pixels
                keep_idx = indices(1); % Keep the first found (top-left priority)
                newWindow(keep_idx) = 1;
            end
            
            % Update the new image with the modified window
            filtered_result(i-1:i+1, j-1:j+1) = newWindow;
        end
    end
end

number_of_points = sum(filtered_result,"all");
% Extract points from newImage (assuming it contains binary minutiae points)
[y, x] = find(filtered_result); % Get coordinates of detected points

figure; % Create a new figure

% First subplot: Original Gray Image
subplot(1,2,1);
imshow(filtered_result); % Display the original gray image
title('Original Gray Image');
hold on; % Keep the image while plotting additional elements

% Plot detected points on the gray image
plot(x, y, 'go', 'MarkerSize', 8, 'LineWidth', 1); % Green circles
legend('Points of Interest'); % Correct legend syntax

% Second subplot: Thinned Binary Image
subplot(1,2,2);
imshow(thin_image_2); % Display the binary image
title('Thinned Image');
hold on; % Keep the image for overlaying points

% Plot the same detected points on the thinned binary image
plot(x, y, 'bo', 'MarkerSize', 8, 'LineWidth', 1); % Green circles
legend('Minutiae'); % Correct legend syntax

hold off; % Release hold for this subplot

number_of_points;
imwrite(filtered_result, filename);