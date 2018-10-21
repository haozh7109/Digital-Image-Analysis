function [ img_GLCM_feature_IDM,img_GLCM_feature_INR,img_GLCM_feature_SHD] = Sliding_Window_GLCM_Analysis(imag,gray_level,dx,dy,windowSize)
% Sliding window GLCM feature analysis
%   using sliding window for GLCM calculation and derive GLCM feature
%   Code by Hao, Oct 12, 2018
%---- input -----
%imag :      the calculated GLCM_Matrix
%GreyLevel:  the grey level used in GLCM calculation

%% parameters setting
%set the window size of GLCM calculation
window_size   = windowSize;
half_win_size = (window_size -1) ./2;

if( mod(window_size(1),2)==0 || mod(window_size(2),2)==0)
    error('-------!!! window size has to be odd number !!! -----')
end

%get the size of input figure
img_input_size = size(imag);

%convert the image from 8 bit(2^8 =256 gray level) to 4 bit (2^4 =16 gray level) and equalize the mosaic image
imag       = histeq(imag,256);
img_edit   = uint8(round(double(imag) * (gray_level-1) / double(max(imag(:)))));

%pad the input image to avoid the edge effect
img_padded = zeros(img_input_size(1)+ 2*half_win_size(1), img_input_size(2)+ 2*half_win_size(2));
img_padded((half_win_size(1)+1):(size(img_padded,1)-half_win_size(1)),((half_win_size(2)+1):(size(img_padded,2)-half_win_size(2)))) = img_edit;

% %define the output
img_GLCM_feature_IDM = zeros(img_input_size);
img_GLCM_feature_INR = zeros(img_input_size);
img_GLCM_feature_SHD = zeros(img_input_size);

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
        
        %select the local image
        imag_loc = img_padded(row_id_pad_beg:row_id_pad_end,col_id_pad_beg:col_id_pad_end);

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
            [glcm_mat_imag_tmp]  =  GLCM_Calculation(imag_loc,gray_level,dx_use,dy_use);
            glcm_mat_imag        =  glcm_mat_imag + glcm_mat_imag_tmp;
        end
        
        %get the average glcm matrix from all directions
        glcm_mat_imag = glcm_mat_imag./directions;
        
        %derive the feature image (value) for GLCM matrix
        [img_GLCM_feature_IDM(irow,icol),img_GLCM_feature_INR(irow,icol),img_GLCM_feature_SHD(irow,icol) ] = GLCM_Feature_Calculation(glcm_mat_imag,gray_level);
    end
end


end

