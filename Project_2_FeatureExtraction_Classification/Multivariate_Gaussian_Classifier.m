function [ class_predict ] = Multivariate_Gaussian_Classifier(TainingFeaturesArray,ClassLabels,TestFeatureArray)
%Function to apply multivariate Gaussian classication
% Code by Hao, Nov 06,2018
%---- input ----
%TainingFeaturesArray   : A 3D matrix (m x n x L) includes L features with feature size: m x n
%ClassLabels            : The predefined class labels
%TestFeatureArray:      : The test feature array (3D matrix with same dimension as TainingFeaturesArray)
%---- output ----
%class_predict: The predicted class labels.

%% datasets for the code test only
if(nargin==0)
    
    test_input    = load('output_project2.mat');
    GLCM_features = test_input.GLCM_features;
    
    %sort the selected features into 3d matrix
    TainingFeaturesArray        = zeros(512,512,3);
    TainingFeaturesArray(:,:,1) = GLCM_features{1,5};
    TainingFeaturesArray(:,:,2) = GLCM_features{1,6};
    TainingFeaturesArray(:,:,3) = GLCM_features{1,8};
    
    %load the predefined mask label
    load('training_mask.mat');
    ClassLabels = training_mask;

end

%% Basic parameters setting

%derive the samples of features(m*n) and feature numbers 
[m,n,L] = size(TainingFeaturesArray);

% derive the number of class (exclude 0)
C = length(unique(ClassLabels))-1; 

%% sort the features from 3d format to 2d format
feature_mat_training = zeros(L,m*n);
feature_mat_test     = zeros(L,m*n);

for iL = 1:L
    tmp1   = TainingFeaturesArray(:,:,iL);
    tmp2   = TestFeatureArray(:,:,iL);
    feature_mat_training(iL,:) = tmp1(:);
    feature_mat_test(iL,:)     = tmp2(:);
end

%% derive the mean of the input feature array for each classes
Mu_s          = zeros(L,C);
ClassLabels   = ClassLabels(:)';

for iC = 1:C
    %get the features matrix belongs to current class
    tmp_feature   = feature_mat_training(:,ClassLabels==iC);
    
    %derive the mean of features
    Mu_s(:,iC) = mean(tmp_feature,2);
end

%% derive the covariance matrix and determinat of feature matrix
cov_mat = cell(1,C); 
det_mat = cell(1,C); 

for iC = 1:C
    %get the features matrix belongs to current class
    tmp_feature   = feature_mat_training(:,ClassLabels==iC);
    
    %get the covariance matrix based on features
    cov_mat{iC} = cov(tmp_feature');
    
    %get the determinant of covariance matrix
    det_mat{iC} = det(cov_mat{iC});
end


%% Apply the classification by multivariate gaussian classifier

%set the output class
class_predict = zeros(m,n);

%apply the classification
%--In this project, we directly use likelyhood to derive the class, since
%the input class has the identical prior probability)--

for im = 1:m
    for in = 1:n
         
        %get the feature vector
        X    = feature_mat_test(:,(in-1)*m+im);
        
        %calculate the probability
        Prob = zeros(1,C);
        for iC = 1:C
            
            if(abs(det_mat{iC}) <= 1e-20) %condition: if the matrix is singular, the determinant is 0
                %calculate the posterior probability (use pseudo inversion in case the covariance matrix is singular)
                Prob(iC) = 1/((2*pi)^(L/2)*sqrt(abs(det_mat{iC}))) * exp(-0.5*(X-Mu_s(:,iC)).' *  pinv(cov_mat{iC}) * (X-Mu_s(:,iC)));
                
            else
                %calculate the posterior probability
                Prob(iC) = 1/((2*pi)^(L/2)*sqrt(abs(det_mat{iC}))) * exp(-0.5*(X-Mu_s(:,iC)).' *  inv(cov_mat{iC}) * (X-Mu_s(:,iC)));
            end
            
        end
        
        %derive the class based on maximal probalibity
        class_predict(im,in) = find(Prob==max(Prob));
        
    end
end

end

