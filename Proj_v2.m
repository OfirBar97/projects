clc;
clear;
% Uncomment the following line to preserve the X-limits of the axes
%xlim(subplot1,[180.972144305487 580.619686783562]);
% Uncomment the following line to preserve the Y-limits of the axes
%ylim(subplot1,[291.331115170728 868.787945615743]);
% Load the image
I = imread('ofir_1.bmp'); % Replace with your image file

max_pixel = max(max(max(I)));
if max_pixel>255
    grayImage = rgb2gray(I); % Convert to grayscale
else 
    grayImage = im2gray(I);
end

subplot(3, 2, 1);
imshow(grayImage);
title('Original Gray Image');
%%
%enhancedImage = fftFingerprintEnhancer(grayImage);
%grayImage=enhancedImage;
%% 

% Creating Average level and Substractin grayscaled image by neighborhood average level

% Initialize variables
[rows, cols] = size(grayImage); % Size of the grayscale image
After_low_I = zeros(rows, cols); % Preallocate for efficiency

% Define the neighborhood size  (windowSize is 'windowSize +1')
windowSize = 11;

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
%%

% Contrast stretching of the 'difference image' using a linear
% transformation 

minIntensity = double(min(differenceImage(:)));
maxIntensity = double(max(differenceImage(:)));

% Perform linear contrast stretching
adjustedImage = (double(differenceImage) - minIntensity) *3*(255 / (maxIntensity - minIntensity));

% Convert back to uint8
adjustedImage = uint8(adjustedImage);

% Display the result
subplot(3, 2, 4);
imshow(adjustedImage);
title('Contrast Adjusted using Linear Stretch');
%%
%enhancedImage = fftFingerprintEnhancer(adjustedImage);
%adjustedImage=enhancedImage;

%%
% Thresholding function -> Binary image


% Define the neighborhood size (i.e., window size)
windowSize = 5; % Adjust as needed
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

% prunnig edges process

%-------------------------------------------------------
% removing white 'edges'

after_binary = after_binary > 0; % making sure '0' or '1's
% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s

% Initialize the result image
resultImage = after_binary;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(after_binary), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 5
% (i.e., the central pixel plus 4 surrounding white pixels)
singleWhitePixels = ((convResult == 4) |(convResult == 5)) & (after_binary == 1);

% Remove the single Black pixels by setting them to White 1
resultImage(singleWhitePixels) = 0;

% Display the results
figure;

subplot(1, 2, 1);
imshow(resultImage);
title('Removed black Pixel edges (branches)');

%%
%-------------------------------------------------------
% removing black 'edges'

% Assuming binaryImage is the binary image (1 for white, 0 for black)
%binaryImage = ~resultImage; % Example image
binaryImage =resultImage;

% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s

% Initialize the result image
resultImage = binaryImage;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 4
% (i.e., the central pixel plus 3 surrounding white pixels)
singleBlackPixels  = ((convResult == 4) | (convResult == 5)) & (after_binary == 1);

% Remove the single white pixels by setting them to black (0)
resultImage(singleBlackPixels) = 0;
 % using 'NOT' to change back the colors

% Display the results
subplot(1, 2, 2);
imshow(resultImage);
title('Removed white Pixel edges (branches)');

%%
% removing white single pixels and 2x2 white cubic pixels
%--------------------------------------------------------


% Define a 4x4 mask for detecting 2x2 white blocks
mask_4x4 = ones(4); % 4x4 block of 1s

% looped 3 times to ensure removal of 2x2 and 1x1 white pixels
for i= 1:3
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

 % inverting the image to use the same algo and invert back

% looped 3 times to ensure riddens of 2x2 and 1x1 white pixels
for i= 1:3
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
    twoByTwoWhitePixels = (convResult_4x4 <= 5)  & (binaryImage == 1);
    
    % Output: Highlight or process 4x4 white blocks as needed
    % Remove 2x2 black blocks by setting them to 0 in the binary image
    binaryImage(twoByTwoWhitePixels) = 0;
    

end  

% Display results

subplot(1, 2,2);
imshow(binaryImage);
title('Removed Single Black Pixels and 2x2 Black Blocks');

%%
%-------------------------------------------------------
% removing white 'edges'

% Initialize the result image
resultImage = binaryImage;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 5
% (i.e., the central pixel plus 4 surrounding white pixels)
singleWhitePixels = (convResult == 4)|(convResult == 5)& (binaryImage == 1);

% Remove the single Black pixels by setting them to White ~(1)
resultImage(singleWhitePixels) = 0;

% Display the results
figure;

subplot(1, 2, 1);
imshow(resultImage);
title('Removed White Pixel edges (branches)');

%%
%-------------------------------------------------------
% removing black 'edges'

% Assuming binaryImage is the binary image (1 for white, 0 for black)
binaryImage = ~resultImage; % Example image

% Initialize the result image
resultImage = binaryImage;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 4
% (i.e., the central pixel plus 3 surrounding white pixels)
singleBlackPixels = (convResult == 4) | (convResult == 5) & (binaryImage == 1);

% Remove the single white pixels by setting them to black (0)
resultImage(singleBlackPixels) = 0;
resultImage = ~resultImage; % using 'NOT' to change back the colors

% Display the results
subplot(1, 2, 2);
imshow(resultImage);
title('Removed black Pixel edges (branches)');
%%
binaryImage=resultImage;
% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); % Perform convolution

% Logical condition: Neighborhood sum is 1, and the original pixel is 1
singleWhitePixels = (convResult == 1) & (binaryImage == 1);

% Remove single white pixels by setting them to 0 in the binary image
binaryImage(singleWhitePixels) = 0;

% using NOT to switch colors and use the same algo
binaryImage= ~binaryImage;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); % Perform convolution

% Logical condition: Neighborhood sum is 1, and the original pixel is 1
singleWhitePixels = (convResult == 1) & (binaryImage == 1);

% Output: Highlight or process single white pixels as needed
% Remove single white pixels by setting them to 0 in the binary image
binaryImage(singleWhitePixels) = 0;


imshow(~binaryImage);
title('Second Black 1x1 & White 1x1 spots cleared Image');
%%

%thin_image=bwmorph(resultImage,'clean',inf);
%thin_image=bwmorph(thin_image,'spur',inf);
%figure;imshow(thin_image);title('spur Image');

%
subplot(1, 2, 1);
thin_image=bwmorph(binaryImage,'clean',inf);
thin_image=~bwmorph(thin_image,'thin',inf);
thin_image=bwmorph(thin_image,'spur',inf);
thin_image=bwareaopen(~thin_image,3);
thin_image=~thin_image;
imshow(thin_image);
title('Thinned Image');


subplot(1, 2, 2);
imshow(grayImage);
title('Original Gray Image');

%%  

% Minutiae extraction
s = size(thin_image); % Get the size of the thinned binary image
N = 3; % Window size (3x3 neighborhood for minutiae detection)
mat = ones(size(thin_image)); % Temporary matrix for storing local neighborhoods
n = (N-1)/2; % Half window size
r = s(1) + 2*n; % New number of rows after padding
c = s(2) + 2*n; % New number of columns after padding

% Initialize temporary padded image and output matrices
double temp(r, c); % Declare temporary image matrix
temp = zeros(r, c); % Initialize with zeros (padding)
bifurcation = zeros(r, c); % Matrix to store bifurcation points
ridge = zeros(r, c); % Matrix to store ridge end points

% Copy the original image into the center of the padded matrix
temp((n+1):(end-n), (n+1):(end-n)) = thin_image(:,:);

% Create an RGB representation of the padded image for visualization
thinned_Img = zeros(r, c, 3); % Initialize 3-channel (RGB) image
thinned_Img(:,:,1) = temp .* 255; % Set red channel
thinned_Img(:,:,2) = temp .* 255; % Set green channel
thinned_Img(:,:,3) = temp .* 255; % Set blue channel

% Sliding window for minutiae detection
for x = (n+1+10):(s(1)+n-10) % Avoid processing near the padded border
    for y = (n+1+10):(s(2)+n-10)
        e = 1; % Row index for 3x3 window
        for k = x-n:x+n
            f = 1; % Column index for 3x3 window
            for l = y-n:y+n
                mat(e, f) = temp(k, l); % Extract 3x3 neighborhood
                f = f + 1;
            end
            e = e + 1;
        end
        % Check if the central pixel is black (ridge pixel)
        if(mat(2, 2) == 0)
            ridge(x, y) = sum(sum(~mat)); % Count white neighbors for ridge endings
            bifurcation(x, y) = sum(sum(~mat)); % Count white neighbors for bifurcations
        end
    end
end

% RIDGE END FINDING
[ridge_x, ridge_y] = find(ridge == 2); % Find coordinates of ridge endings
len = length(ridge_x); % Number of ridge endings
% For Display (mark ridge endings with green squares)
for i = 1:len
    % Clear area around ridge ending (set to black)
    thinned_Img((ridge_x(i)-3):(ridge_x(i)+3), (ridge_y(i)-3), 1:3) = 0;
    thinned_Img((ridge_x(i)-3):(ridge_x(i)+3), (ridge_y(i)+3), 1:3) = 0;
    thinned_Img((ridge_x(i)-3), (ridge_y(i)-3):(ridge_y(i)+3), 1:3) = 0;
    thinned_Img((ridge_x(i)+3), (ridge_y(i)-3):(ridge_y(i)+3), 1:3) = 0;

    % Mark ridge ending in green
    thinned_Img((ridge_x(i)-3):(ridge_x(i)+3), (ridge_y(i)-3), 2) = 255;
    thinned_Img((ridge_x(i)-3):(ridge_x(i)+3), (ridge_y(i)+3), 2) = 255;
    thinned_Img((ridge_x(i)-3), (ridge_y(i)-3):(ridge_y(i)+3), 2) = 255;
    thinned_Img((ridge_x(i)+3), (ridge_y(i)-3):(ridge_y(i)+3), 2) = 255;
end

% BIFURCATION FINDING
[bifurcation_x, bifurcation_y] = find(bifurcation == 4); % Find coordinates of bifurcations
len = length(bifurcation_x); % Number of bifurcations
% For Display (mark bifurcations with blue squares)
for i = 1:len
    % Clear area around bifurcation (set to black)
    thinned_Img((bifurcation_x(i)-3):(bifurcation_x(i)+3), (bifurcation_y(i)-3), 1:2) = 0;
    thinned_Img((bifurcation_x(i)-3):(bifurcation_x(i)+3), (bifurcation_y(i)+3), 1:2) = 0;
    thinned_Img((bifurcation_x(i)-3), (bifurcation_y(i)-3):(bifurcation_y(i)+3), 1:2) = 0;
    thinned_Img((bifurcation_x(i)+3), (bifurcation_y(i)-3):(bifurcation_y(i)+3), 1:2) = 0;
end

figure;
subplot(1,3,1)
imshow(thinned_Img);
title('Minutiae');
hold on;
% Dummy markers for legend
plot(NaN, NaN, 'sr', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Red square for ridge endings
plot(NaN, NaN, 'sb', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); % Blue square for bifurcations
legend({'Ridges End', 'Bifurcation'}, 'Location', 'northeast', 'FontSize', 10);
hold off;

subplot(1,3,2)
imshow(grayImage);
title('Original Gray Image');
%%
% trying to extract useful information, using extraction of x,y
% coordiantes, then transforming into a 2D list, and number of points.
% Matrix  A = first_register of image;
% Matrix  B = second_image for validation;
% C = number of A[x,y] that exists in B[x,y]
% D = number of B[x,y] that exists in A[x,y]
% NB = number of elements in  B, NA = number of elements in  A
% calculation -> C+D/(A+B) *100 = percentage of occurences matching
% Initialize lists for ridge endings (A) and bifurcations (B)

%A=[];

% Get image size
[rows, cols] = size(thin_image);

% Initialize result matrix with zeros
result = zeros(rows, cols);

% Loop through the image, avoiding the borders
for i = 10:rows-10
    for j = 10:cols-10
        if thin_image(i, j) == 0  % Check only ridge points
            % Extract 3x3 neighborhood
            neighborhood = thin_image(i-1:i+1, j-1:j+1);
            
            % Count black pixels (zeros)
            black_pixel_count = sum(neighborhood(:));
            
            % Apply the conditions
            if black_pixel_count == 7
                result(i, j) = 1;  % Ridge ending
                %A = [A; i,j];
                
            elseif black_pixel_count ==  5
                result(i, j) = 1;  % Bifurcation
                %A = [A; i,j];
            else

            end
        end
    end
end

number_of_points = sum(result,"all");
% Save result matrix
save('result.mat', 'result');
subplot(1,3,3)
imshow(grayImage);
title('Original Gray Image');
imshow(result);
title("points of interest");
number_of_points;

