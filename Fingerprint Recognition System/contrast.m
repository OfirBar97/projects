function Contrast = contrast(Average_level)
% Contrast stretching of the 'difference image' using a linear
% transformation 

minIntensity = double(min(Average_level(:)));
maxIntensity = double(max(Average_level(:)));

% Perform linear contrast stretching
adjustedImage = (double(Average_level) - minIntensity) *3*(255 / (maxIntensity - minIntensity));

% Convert back to uint8
Contrast = uint8(adjustedImage);

% Display the result
subplot(3, 2, 4);
imshow(Contrast);
title('Contrast Adjusted using Linear Stretch');