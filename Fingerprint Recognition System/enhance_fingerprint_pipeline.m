function enhancedImage = enhance_fingerprint_pipeline(img)
% ENHANCE_FINGERPRINT_PIPELINE performs fingerprint enhancement in stages:
% normalization, orientation, frequency estimation, and Gabor filtering.

    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = double(img);

    %% Stage 1: Normalize
    normalized = normalize_image(img, 100, 80);
imshow(normalized)
    %% Stage 2: Orientation Field
    blockSize = 64;
    orientation = compute_orientation(normalized, blockSize);

    %% Stage 3: Frequency Estimation
    frequency = estimate_frequency(normalized, orientation, blockSize);

    %% Stage 4: Gabor Filtering
    enhancedImage = gabor_enhance(normalized, orientation, frequency, blockSize);

    %% Display result
    figure;
    subplot(1, 2, 1); imshow(img, []); title('Original');
    subplot(1, 2, 2); imshow(enhancedImage, []); title('Enhanced');
end

function normImage = normalize_image(img, M0, VAR0)
    M = mean(img(:));
    VAR = var(img(:));
    normImage = sqrt(VAR0 * ((img - M).^2) / VAR) .* sign(img - M) + M0;
end

function orientation = compute_orientation(img, blockSize)
    [Gx, Gy] = imgradientxy(img);
    [rows, cols] = size(img);
    orientation = zeros(floor(rows / blockSize), floor(cols / blockSize));
    for i = 1:blockSize:rows - blockSize + 1
        for j = 1:blockSize:cols - blockSize + 1
            blockGx = Gx(i:i+blockSize-1, j:j+blockSize-1);
            blockGy = Gy(i:i+blockSize-1, j:j+blockSize-1);
            Vx = sum(sum(2 * blockGx .* blockGy));
            Vy = sum(sum(blockGx.^2 - blockGy.^2));
            orientation(ceil(i / blockSize), ceil(j / blockSize)) = 0.5 * atan2(Vx, Vy);
        end
    end
end

function frequencyMap = estimate_frequency(img, orientation, blockSize)
    [h, w] = size(orientation);
    frequencyMap = ones(h, w) * 1/10; % Placeholder for actual frequency estimation
end

function enhanced = gabor_enhance(img, orientation, frequency, blockSize)
    [h, w] = size(img);
    enhanced = zeros(h, w);
    for i = 1:blockSize:h - blockSize + 1
        for j = 1:blockSize:w - blockSize + 1
            block = img(i:i+blockSize-1, j:j+blockSize-1);
            ang = orientation(ceil(i / blockSize), ceil(j / blockSize));
            freq = frequency(ceil(i / blockSize), ceil(j / blockSize));
            lambda = 1 / freq;
            gabor = gabor_fn(4, ang, lambda, 0.5, 0);
            filtered = imfilter(block, gabor, 'symmetric');
            enhanced(i:i+blockSize-1, j:j+blockSize-1) = filtered;
        end
    end
end

function g = gabor_fn(sigma, theta, lambda, gamma, psi)
    sigma_x = sigma;
    sigma_y = sigma / gamma;
    sz = fix(6 * sigma);
    [x, y] = meshgrid(-sz:sz, -sz:sz);
    x_theta = x * cos(theta) + y * sin(theta);
    y_theta = -x * sin(theta) + y * cos(theta);
    g = exp(-0.5 * (x_theta.^2 / sigma_x^2 + y_theta.^2 / sigma_y^2)) ...
        .* cos(2 * pi * x_theta / lambda + psi);
end