function pre_thinned = remove_noise_blobs(pruned_image)
% REMOVE_NOISE_BLOBS removes isolated pixels and 2x2 pixel blocks
% from a binary image. First removes white noise, then black noise.
%
% Input:
%   pruned_image - binary image (logical or 0/1)
%
% Output:
%   pre_thinned - binary image after noise removal

% Initialize
binaryImage = pruned_image;

% Define convolution masks
mask = ones(3);
mask_4x4 = ones(4);

% --- Stage 1: Remove white single pixels and 2x2 white blocks ---
for i = 1:4
    % Remove isolated white pixels
    convResult = conv2(double(binaryImage), mask, 'same');
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
    binaryImage(singleWhitePixels) = 0;

    % Remove 2x2 white blocks (approximate by <= 4 sum in 4x4)
    convResult_4x4 = conv2(double(binaryImage), mask_4x4, 'same');
    twoByTwoWhitePixels = (convResult_4x4 <= 4) & (binaryImage == 1);
    binaryImage(twoByTwoWhitePixels) = 0;
end

% --- Stage 2: Remove black single pixels and 2x2 black blocks ---
binaryImage = ~binaryImage; % invert image to process black pixels as white

for i = 1:2
    % Remove isolated black (now white) pixels
    convResult = conv2(double(binaryImage), mask, 'same');
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
    binaryImage(singleWhitePixels) = 0;

    % Remove 2x2 black (now white) blocks
    convResult_4x4 = conv2(double(binaryImage), mask_4x4, 'same');
    twoByTwoWhitePixels = (convResult_4x4 <= 4) & (binaryImage == 1);
    binaryImage(twoByTwoWhitePixels) = 0;
end

% Invert back to original polarity
pre_thinned = ~binaryImage;
end
