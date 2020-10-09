%%=======================================================================
%% Matlab code for project-1
% Author: Hao, Zhao
% Date:   2018, Oct 08
%%=======================================================================

%clean the Matlab enviroment
close all; clear all; clc

%% PartA: Loading the images and seperate into individual textures

% load the two mosic images
img1 = imread('mosaic1.png');
img2 = imread('mosaic2.png');

% display the input raw images
figure;imagesc(img1);
colormap gray;
colorbar;
title('=== masaic image-1 ===')

figure;imagesc(img2);
colormap gray;
colorbar;
title('=== masaic image-2 ===')

% seperate the initial images into 8 individual masic images
texture_1 = img1(1:256,1:256);
texture_2 = img1(257:512,1:256);
texture_3 = img1(1:256,257:512);
texture_4 = img1(257:512,257:512);
texture_5 = img2(1:256,1:256);
texture_6 = img2(257:512,1:256);
texture_7 = img2(1:256,257:512);
texture_8 = img2(257:512,257:512);

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


% calculate the GLCM for each textures with dx=1, and dy=0

gray_level = 16;
dx = 1;
dy = 0;

for n = 1:8
    
    %calculate the Normalized GLCM for each textutre figures
    [glcm_mat]  = GLCM_Calculation(texture_all_new{n},gray_level,dx,dy);
    
    figure;
    suptitle(sprintf(['=== Equalized mosaic image and corresponding GLCM: ' num2str(n) ' === \n']))
    
    subplot(1,2,1)
    imagesc(texture_all_new{n});
    colormap gray;
    colorbar;
    title('Equalized image')

    subplot(1,2,2)
    imagesc(glcm_mat);
    colormap Parula;
    colorbar;
    xlim([0,15]);
    title('Normalized GLCM matrix')
    
end


%% Part C: Computing GLCM feature images in local windows

%set the window size of GLCM calculation
window_size   = [31,31];
half_win_size = (window_size -1) ./2; 

%get the size of input figure
img_input_size = size(img1);

%convert the image from 8 bit(2^8 =256 gray level) to 4 bit (2^4 =16 gray level) and equalize the mosaic image
gray_level = 2^4;
img1_edit = uint8(round(double(img1) * (gray_level-1) / double(max(img1(:)))));
img2_edit = uint8(round(double(img2) * (gray_level-1) / double(max(img2(:)))));

%pad the input image to avoid the edge effect
img1_padded = zeros(img_input_size(1)+ 2*half_win_size(1), img_input_size(2)+ 2*half_win_size(2));
img2_padded = zeros(img_input_size(1)+ 2*half_win_size(1), img_input_size(2)+ 2*half_win_size(2));
img1_padded((half_win_size(1)+1):(size(img1_padded,1)-half_win_size(1)),((half_win_size(2)+1):(size(img1_padded,2)-half_win_size(2)))) = img1_edit;
img2_padded((half_win_size(1)+1):(size(img2_padded,1)-half_win_size(1)),((half_win_size(2)+1):(size(img2_padded,2)-half_win_size(2)))) = img2_edit;

% %define the output 
img1_GLCM_feature_IDM = zeros(img_input_size);
img1_GLCM_feature_INR = zeros(img_input_size);
img1_GLCM_feature_SHD = zeros(img_input_size);

img2_GLCM_feature_IDM = zeros(img_input_size);
img2_GLCM_feature_INR = zeros(img_input_size);
img2_GLCM_feature_SHD = zeros(img_input_size);

%derive the GLCM feature by sliding window
for irow = 1:img_input_size(1)
    %display status
    fprintf('=========== running on row : %d of %d ==============\n',irow,img_input_size(1));
    for icol = 1:img_input_size(2)
        
    %select the image for GLCM calculation
    win_cent_row   = irow + half_win_size(1);
    win_cent_col   = icol + half_win_size(2);
    row_id_pad_beg = (win_cent_row) - half_win_size(1);
    row_id_pad_end = (win_cent_row) + half_win_size(1);
    col_id_pad_beg = (win_cent_col) - half_win_size(2);
    col_id_pad_end = (win_cent_col) + half_win_size(2);
    imag1_loc = img1_padded(row_id_pad_beg:row_id_pad_end,col_id_pad_beg:col_id_pad_end);
    imag2_loc = img2_padded(row_id_pad_beg:row_id_pad_end,col_id_pad_beg:col_id_pad_end);
    
    % calculate the GLCM for each textures with dx=1, and dy=0
    gray_level = 16;
    dx = 1;
    dy = 1;
    
    %derive the GLCM matrix
    [glcm_mat_imag1]  = GLCM_Calculation(imag1_loc,gray_level,dx,dy);
    [glcm_mat_imag2]  = GLCM_Calculation(imag2_loc,gray_level,dx,dy);
    
    %derive the feature image (value) for GLCM matrix
    [img1_GLCM_feature_IDM(irow,icol),img1_GLCM_feature_INR(irow,icol),img1_GLCM_feature_SHD(irow,icol) ] = GLCM_Feature_Calculation(glcm_mat_imag1,gray_level);
    [img2_GLCM_feature_IDM(irow,icol),img2_GLCM_feature_INR(irow,icol),img2_GLCM_feature_SHD(irow,icol) ] = GLCM_Feature_Calculation(glcm_mat_imag2,gray_level);
    
    end
end

% display the derived GLCM feature matrix
figure;
subplot(2,2,1)
imagesc(img1_edit);
colorbar;
title('input Image')
subplot(2,2,2)
imagesc(img1_GLCM_feature_IDM);
colorbar;
title('IDM')
subplot(2,2,3)
imagesc(img1_GLCM_feature_INR);
colorbar;
title('INR')
subplot(2,2,4)
imagesc(img1_GLCM_feature_SHD);
colorbar;
title('SHD')

imagesc(img2_GLCM_feature_IDM);
figure;title('IDM')
imagesc(img2_GLCM_feature_INR);
figure;title('INR')
imagesc(img2_GLCM_feature_SHD);
figure;title('SHD')



