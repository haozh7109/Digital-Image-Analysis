%%=======================================================================
%% Matlab code for project-1
% Author: Hao, Zhao
% Date:   2018, Oct 08
%%=======================================================================

%clean the Matlab enviroment
close all; clear all; clc

%% PartA: Loading the images and seperate into individual textures

% load the two mosic images
mosaic1 = imread('mosaic1.png');
mosaic2 = imread('mosaic2.png');

% display the input raw images
figure;imagesc(mosaic1);
colormap gray;
colorbar;
title('=== masaic image-1 ===')

figure;imagesc(mosaic2);
colormap gray;
colorbar;
title('=== masaic image-2 ===')

% seperate the initial images into 8 individual masic images
texture_1 = mosaic1(1:256,1:256);
texture_2 = mosaic1(257:512,1:256);
texture_3 = mosaic1(1:256,257:512);
texture_4 = mosaic1(257:512,257:512);
texture_5 = mosaic2(1:256,1:256);
texture_6 = mosaic2(257:512,1:256);
texture_7 = mosaic2(1:256,257:512);
texture_8 = mosaic2(257:512,257:512);

texture_all    = cell(1,8);
texture_all{1} = texture_1;
texture_all{2} = texture_2;
texture_all{3} = texture_3;
texture_all{4} = texture_4;
texture_all{5} = texture_5;
texture_all{6} = texture_6;
texture_all{7} = texture_7;
texture_all{8} = texture_8;

%% PartB: analysis the 2nd order statistic parameters(GLCM) for individual image

% Equaliz the mosaic images and display 
texture_all_eq     = cell(1,8);
texture_all_new    = cell(1,8);

for n = 1:8
    
    %convert the image from 8 bit(2^8 =256 gray level) to 4 bit (2^4 =16 gray level) and equalize the mosaic image
    gray_level         = 2^4;
    texture_all_eq{n}  = histeq(texture_all{n},256);
    texture_all_new{n} = uint8(round(double(texture_all_eq{n}) * (gray_level-1) / double(max(texture_all_eq{n}(:)))));
    
    figure;
    suptitle(sprintf(['=== Texture image: ' num2str(n) ' === \n']))
    
    subplot(2,3,1)
    imagesc(texture_all{n});
    colormap gray;
    colorbar;
    title('Original image')
    
    subplot(2,3,2)
    imagesc(texture_all_eq{n});
    colormap gray;
    colorbar;
    title('Equalized image')
    
    subplot(2,3,3)
    imagesc(texture_all_new{n});
    colormap gray;
    colorbar;
    title('Equalized image with 16 gray level')
    
    subplot(2,3,4)
    imhist(texture_all{n});
    xlim([0,255])
    title('Hisogram of original image')
    
    subplot(2,3,5)
    imhist(texture_all_eq{n});
    xlim([0,255])
    title('Hisogram of equalized image')
    
    subplot(2,3,6)
    imhist(texture_all_new{n});
    xlim([0,15])
    title('Hisogram of equalized image with 16 gray level')
    
end


% calculate the GLCM for each textures
for n = 1:8
    
    %define the gray level for GLCM analysis
    gray_level = 16;
    
    %define the GLCM analyzing parameters (step and angle)
    switch n
        case 1 % isotropic GLCM
            dx = [1, 1, 0, -1];
            dy = [0,-1,-1, -1];
        case 2 % 90 degree GLCM
            dx = 1;
            dy = 0;
        case 3 % 0 and 90 degree GLCM
            dx = [1,0];
            dy = [0,1];
        case 4 % 0 and 90 degree GLCM
            dx = [1,0];
            dy = [0,1];
        case 5  % 45 and 135 degree GLCM
            dx = [1,1];
            dy = [1,-1];
        case 6  % 90 degree GLCM
            dx = 1;
            dy = 0;
        case 7 % 0 and 90 degree GLCM
            dx = [1,0];
            dy = [0,1];
        otherwise % isotropic GLCM
            dx = [1, 1, 0, -1];
            dy = [0,-1,-1, -1];
    end
    
    %derive the GLCM matrix by selected direction
    if(length(dx) ~= length(dy))
        error('------!!! dx and dy must have same size !!! --------')
    end
    
    % derive the glcm matrix based on defined mulit/single direction
    directions    = length(dx);
    glcm_mat_imag = 0;
    
    for itheta = 1:directions
        dx_use = dx(itheta);
        dy_use = dy(itheta);
        [glcm_mat_imag_tmp]  =  GLCM_Calculation(texture_all_new{n},gray_level,dx_use,dy_use);
        glcm_mat_imag        =  glcm_mat_imag + glcm_mat_imag_tmp;
    end
    
    %get the average glcm matrix from all directions
    glcm_mat_imag = glcm_mat_imag./directions;

    figure;
    suptitle(sprintf(['=== Equalized texture image and corresponding GLCM: ' num2str(n) ' === \n']))
    
    subplot(1,2,1)
    imagesc(texture_all_new{n});
    colormap gray;
    colorbar;
    title('Equalized image')

    subplot(1,2,2)
    imagesc(glcm_mat_imag);
    colormap Parula;
    colorbar;
    xlim([0,15]);
    title('Normalized GLCM matrix')
    
end


%% Part C: Computing GLCM feature images in local windows

%convert the image from 8 bit(2^8 =256 gray level) to 4 bit (2^4 =16 gray level) and equalize the mosaic image
gray_level  = 2^4;
mosaic1_eq  = histeq(mosaic1,256);
mosaic1_new = uint8(round(double(mosaic1_eq) * (gray_level-1) / double(max(mosaic1_eq(:)))));
mosaic2_eq  = histeq(mosaic2,256);
mosaic2_new = uint8(round(double(mosaic2_eq) * (gray_level-1) / double(max(mosaic2_eq(:)))));

%set window and gray parammeters for sliding window GLCM analysis:
window     = [31,31];
gray_level = 16;

%save the derived features for segmentation use
GLCM_features = cell(3,8);

for n = 1:8
    
    %define the GLCM analyzing parameters (step and angle)
    switch n
        case 1 % isotropic GLCM
            dx    = [1, 1, 0, -1];
            dy    = [0,-1,-1, -1];
            image = mosaic1_new;
        case 2 % 90 degree GLCM
            dx = 1;
            dy = 0;
            image = mosaic1_new;
        case 3 % 0 and 90 degree GLCM
            dx = [1,0];
            dy = [0,1];
            image = mosaic1_new;
        case 4 % 0 and 90 degree GLCM
            dx = [1,0];
            dy = [0,1];
            image = mosaic1_new;
        case 5  % 45 and 135 degree GLCM
            dx = [1,1];
            dy = [1,-1];
            image = mosaic2_new;
        case 6  % 90 degree GLCM
            dx = 1;
            dy = 0;
            image = mosaic2_new;
        case 7 % 0 and 90 degree GLCM
            dx = [1,0];
            dy = [0,1];
            image = mosaic2_new;
        otherwise % isotropic GLCM
            dx = [1, 1, 0, -1];
            dy = [0,-1,-1, -1];
            image = mosaic2_new;
    end
    
    %start sliding window GLCM calculation
    [img_GLCM_feature_IDM,img_GLCM_feature_INR,img_GLCM_feature_SHD] = Sliding_Window_GLCM_Analysis(image,gray_level,dx,dy,window);
    
    % display the derived GLCM feature matrix
    figure;
    subplot(2,2,1)
    imagesc(image);
    colorbar;
    title('input Image')
    subplot(2,2,2)
    imagesc(img_GLCM_feature_IDM);
    colorbar;
    title('IDM')
    subplot(2,2,3)
    imagesc(img_GLCM_feature_INR);
    colorbar;
    title('INR')
    subplot(2,2,4)
    imagesc(img_GLCM_feature_SHD);
    colorbar;
    title('SHD')
    
    % save the derived GLCM features for segmentation 
    GLCM_features{1,n} = img_GLCM_feature_IDM;
    GLCM_features{2,n} = img_GLCM_feature_INR;
    GLCM_features{3,n} = img_GLCM_feature_SHD;
end

%% Part D: seperate the textures by using GLCM feature thresholding

% display the thresholded feature for texture 1
img_GLCM_feature_IDM = GLCM_features{1,1};
img_GLCM_feature_INR = GLCM_features{2,1};
img_GLCM_feature_SHD = GLCM_features{3,1};

threshould_imag =  mosaic1_new;
threshould_imag(abs(img_GLCM_feature_INR)<10 | abs(img_GLCM_feature_INR)>12) = nan;

figure;
suptitle('=== Segmentation of texture 1 ===')
subplot(1,2,1)
imagesc(image);
colorbar;
subplot(1,2,2)
imagesc(threshould_imag);
colorbar;

% display the thresholded feature for texture 2
img_GLCM_feature_IDM = GLCM_features{1,2};
img_GLCM_feature_INR = GLCM_features{2,2};
img_GLCM_feature_SHD = GLCM_features{3,2};
threshould_imag =  mosaic1_new;
threshould_imag(abs(img_GLCM_feature_IDM)<0.55) = nan;

figure;
suptitle('=== Segmentation of texture 2 ===')
subplot(1,2,1)
imagesc(image);
colorbar;
subplot(1,2,2)
imagesc(threshould_imag);
colorbar;

% display the thresholded feature for texture 3
img_GLCM_feature_IDM = GLCM_features{1,3};
img_GLCM_feature_INR = GLCM_features{2,3};
img_GLCM_feature_SHD = GLCM_features{3,3};
threshould_imag =  mosaic1_new;
threshould_imag(abs(img_GLCM_feature_INR)>5) = nan;

figure;
suptitle('=== Segmentation of texture 3 ===')
subplot(1,2,1)
imagesc(image);
colorbar;
subplot(1,2,2)
imagesc(threshould_imag);
colorbar;

% display the thresholded feature for texture 4
img_GLCM_feature_IDM = GLCM_features{1,4};
img_GLCM_feature_INR = GLCM_features{2,4};
img_GLCM_feature_SHD = GLCM_features{3,4};
threshould_imag =  mosaic1_new;
threshould_imag(abs(img_GLCM_feature_IDM)>0.4) = nan;

figure;
suptitle('=== Segmentation of texture 4 ===')
subplot(1,2,1)
imagesc(image);
colorbar;
subplot(1,2,2)
imagesc(threshould_imag);
colorbar;


% display the thresholded feature for texture 5
img_GLCM_feature_IDM = GLCM_features{1,5};
img_GLCM_feature_INR = GLCM_features{2,5};
img_GLCM_feature_SHD = GLCM_features{3,5};
threshould_imag =  mosaic2_new;
threshould_imag(abs(img_GLCM_feature_INR)<=15) = nan;

figure;
suptitle('=== Segmentation of texture 5 ===')
subplot(1,2,1)
imagesc(image);
colorbar;
subplot(1,2,2)
imagesc(threshould_imag);
colorbar;


% display the thresholded feature for texture 6
img_GLCM_feature_IDM = GLCM_features{1,6};
img_GLCM_feature_INR = GLCM_features{2,6};
img_GLCM_feature_SHD = GLCM_features{3,6};
threshould_imag =  mosaic2_new;
threshould_imag(abs(img_GLCM_feature_INR)>5) = nan;

figure;
suptitle('=== Segmentation of texture 6 ===')
subplot(1,2,1)
imagesc(image);
colorbar;
subplot(1,2,2)
imagesc(threshould_imag);
colorbar;

% display the thresholded feature for texture 7
img_GLCM_feature_IDM = GLCM_features{1,7};
img_GLCM_feature_INR = GLCM_features{2,7};
img_GLCM_feature_SHD = GLCM_features{3,7};
threshould_imag =  mosaic2_new;
threshould_imag(abs(img_GLCM_feature_SHD)<6000) = nan;

figure;
suptitle('=== Segmentation of texture 7 ===')
subplot(1,2,1)
imagesc(image);
colorbar;
subplot(1,2,2)
imagesc(threshould_imag);
colorbar;


% display the thresholded feature for texture 8
img_GLCM_feature_IDM = GLCM_features{1,8};
img_GLCM_feature_INR = GLCM_features{2,8};
img_GLCM_feature_SHD = GLCM_features{3,8};
threshould_imag =  mosaic2_new;
threshould_imag(abs(img_GLCM_feature_INR)>10 & abs(img_GLCM_feature_INR)<12) = nan;

figure;
suptitle('=== Segmentation of texture 8 ===')
subplot(1,2,1)
imagesc(image);
colorbar;
subplot(1,2,2)
imagesc(threshould_imag);
colorbar;