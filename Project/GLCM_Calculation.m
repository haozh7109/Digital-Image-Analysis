function [ glcm_mat] = GLCM_Calculation(imag,GrayLevel,dx,dy)

%Gray Level Co-ocurrence Matrix calculation
%   Generate the GLCM for an selected image 
%   Code by Hao, Oct 05, 2018
%---- input -----
%imag :      the image for the GLCM calculation
%GreyLevel:  the gray level of input image
%dx,dy:      the dislocation of reference point and seeking point 


%% get the size of input image
[row,col] = size(imag);

%% define the output of GLCM matrix
glcm_mat     = zeros(GrayLevel,GrayLevel);
glcm_mat_ind = 0:1:GrayLevel-1;

% Loop through all the input image location to generate the GLCM response
for i = 1:row
    for j = 1:col
        
        %define the reference point 
        GrayLevel_ref = imag(i,j);
        
        %check whether if the the desinged point is out off image bounday 
        if i+dy > row || i+dy < 1 || j+dx > col || j+dy < 1 || i + dx < 1 || j + dx < 1
            continue
        else
            %locate the point in the designed dislocation
            GrayLevel_next = imag(i+dy,j+dx);
            
            %insert the gray level response into the GLCM matrix
            glcm_row_indx = find(glcm_mat_ind==GrayLevel_ref);
            glcm_col_indx = find(glcm_mat_ind==GrayLevel_next);
            glcm_mat(glcm_row_indx, glcm_col_indx) = glcm_mat(glcm_row_indx, glcm_col_indx) + 1;
            
        end
    end
end


% generate the symmetric GLCM matrix by adding it's transpose 
glcm_mat = glcm_mat+glcm_mat';

% generate the probability (normalized version of glcm)
glcm_mat = glcm_mat/sum(sum(glcm_mat));


end

