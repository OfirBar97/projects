function cleaned_image = remove_black_edges(after_binary)
% REMOVE_BLACK_EDGES removes thin or isolated black artifacts
% (e.g., "branches") from a binary image.
%
% Input:
%   after_binary - binary image (logical or 0/1)
%
% Output:
%   cleaned_image - binary image after black edge removal

% Ensure input is logical
after_binary = after_binary > 0;

% Define a 3x3 mask
mask = ones(3);

% Initialize result
resultImage = after_binary;

% Apply the black edge removal for 3 iterations
for j = 1:1
    convResult = conv2(double(resultImage), mask, 'same');
    singleBlackPixels = ((convResult == 4) | (convResult == 5)) & (resultImage == 0);
    resultImage(singleBlackPixels) = 1;
end

% Final cleanup iteration
convResult = conv2(double(resultImage), mask, 'same');
singleBlackPixels = ((convResult == 4) | (convResult == 5)) & (resultImage == 0);
resultImage(singleBlackPixels) = 1;

% Output
cleaned_image = resultImage;
end
