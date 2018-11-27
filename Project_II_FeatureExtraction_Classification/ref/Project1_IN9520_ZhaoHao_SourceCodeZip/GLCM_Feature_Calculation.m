function [ IDM,INR,SHD ] = GLCM_Feature_Calculation(GLCM_Matrix,GrayLevel)
%Calculate the GLCM feature based on inpout GLCM matrix
%   Generate the GLCM feature for calculated GLCM matrix
%   Code by Hao, Oct 11, 2018
%---- input -----
%imag :      the calculated GLCM_Matrix
%GreyLevel:  the grey level used in GLCM calculation


%% parameters setting
%define the probability
p = GLCM_Matrix;

%define the i(row) and j(column) index matrix
i = repmat((0:(GrayLevel-1))', 1, GrayLevel); % column vector i 
j = repmat( 0:(GrayLevel-1)  , GrayLevel, 1); % row vector j

%% generate the feature image (value) for current GLCM matrix
%(1) generate Inverse DIfference Moment(IDM) value for current GLCM matrix
weight_IDM  = 1./(1+(i-j).^2);
IDM         = sum(sum(weight_IDM .* p));
%(2) generate Inertia/Contrast value for current GLCM matrix
weight_INR  = (i-j).^2;
INR         = sum(sum(weight_INR .* p));
%(3) generate Cluster shade for current GLCM matrix
Px_i        = sum(p,2);
Py_j        = sum(p,1);
Ux          = sum(i(:,1).* Px_i);
Uy          = sum(j(1,:).* Py_j);
weight_SHD  = (i-j-Ux-Uy).^3;
SHD         = sum(sum(weight_SHD .* p));




end

