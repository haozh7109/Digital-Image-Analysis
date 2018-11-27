function [ Quadrant_features ] = GLCM_Feature_Calculation_Quadrants(GLCM_Matrix,GrayLevel)
%Calculate the GLCM feature based on inpout GLCM matrix
%   Generate the GLCM feature for calculated GLCM matrix
%   Code by Hao, Nov 04, 2018
%---- input -----
%imag :      the calculated GLCM_Matrix
%GreyLevel:  the grey level used in GLCM calculation


%% parameters setting
if(GrayLevel~=16)
    error('!!! the gray level is set to 16 by default in quadrant features caluculation !!!')
end

%% generate the feature image (value) for current GLCM matrix
Quadrant_features = nan(1,8);

%get the features for 4 quadrants
Quadrant_features(1) = sum(sum(GLCM_Matrix(1:8,1:8)))/sum(GLCM_Matrix(:));
Quadrant_features(2) = sum(sum(GLCM_Matrix(1:8,9:16)))/sum(GLCM_Matrix(:));
Quadrant_features(3) = sum(sum(GLCM_Matrix(9:16,1:8)))/sum(GLCM_Matrix(:));
Quadrant_features(4) = sum(sum(GLCM_Matrix(9:16,9:16)))/sum(GLCM_Matrix(:));

%derive the sub-quadrant features from above 1st quadrant
Quadrant_features(5) = sum(sum(GLCM_Matrix(1:4,1:4)))/sum(GLCM_Matrix(:));
Quadrant_features(6) = sum(sum(GLCM_Matrix(1:4,5:8)))/sum(GLCM_Matrix(:));
Quadrant_features(7) = sum(sum(GLCM_Matrix(5:8,1:4)))/sum(GLCM_Matrix(:));
Quadrant_features(8) = sum(sum(GLCM_Matrix(5:8,5:8)))/sum(GLCM_Matrix(:));

end

