function Loaded_image = image_read(image_path)
I = imread(image_path); % image file

max_pixel = max(max(max(I)));
if max_pixel>255
    Loaded_image = rgb2gray(I); % Convert to grayscale
else 
    Loaded_image = im2gray(I);
end
%grayImage=grayImage(94:410,67:234); % fp1 zoom or fp3,,
subplot(3, 2, 1);
imshow(Loaded_image);
title('Original Gray Image');
end