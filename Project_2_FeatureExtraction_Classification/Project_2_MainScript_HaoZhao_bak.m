%%======================================================================
% Main Script for Image Analysis Course, Mandatory Project-2
% Author: Hao Zhao
% Date:   Nov 1, 2018
%%======================================================================


%% set the working path and enviroment
data_path = 'C:\Users\haozh\Desktop\ImageAnalysis_Assignment2\oblig2';
addpath(genpath(data_path));
close all;clear;clc;

%% Part-1: Choosing GLCM images to work with

%---load the testing images and corresponding GLCM matrices---
load('mosaic1_train.mat');
load('mosaic2_test.mat');
load('mosaic3_test.mat');

%display the training and testing datasets
figure;imagesc(mosaic1_train);title('Training dataset: mosaic1')
figure;imagesc(mosaic2_test); title('Testing dataset: mosaic2')
figure;imagesc(mosaic3_test); title('Testing dataset: mosaic3')

%load the predefined mask label
load('training_mask.mat');
ClassLabels = training_mask;

%display the predifined class lable
figure;imagesc(ClassLabels);title('Training Class Label')

%---load the GLCM matrices---
for idata = 1:4
    
    %load the GLCM for current texture
    load(['texture' num2str(idata) 'dxplus1dy0']);
    load(['texture' num2str(idata) 'dxplus1dymin1']);
    load(['texture' num2str(idata) 'dx0dymin1']);
    load(['texture' num2str(idata) 'dxmin1dymin1']);

    %display the current texture and GLCM
    if(idata==1)
        
        figure;
        suptitle(sprintf('=== Texture: %d ===\n',idata));
        subplot(2,3,1);imagesc(mosaic1_train(1:256,1:256));title('Input Image');colorbar
        subplot(2,3,2);imagesc(texture1dx1dy0); title('GLCM: 0   degree ');colorbar
        subplot(2,3,3);imagesc(texture1dx1dymin1);title('GLCM: 45  degree ');colorbar
        subplot(2,3,4);imagesc(texture1dx0dymin1);title('GLCM: 90  degree ');colorbar
        subplot(2,3,5);imagesc(texture1dxmin1dymin1);title('GLCM: 135 degree ');colorbar
        subplot(2,3,6);imagesc((texture1dx1dy0+texture1dx0dymin1)*0.5);title('GLCM: 0 + 90 degree ');colorbar
        
    elseif(idata==2)

        figure;
        suptitle(sprintf('=== Texture: %d ===\n',idata));
        subplot(2,3,1);imagesc(mosaic1_train(1:256,257:512));title('Input Image');colorbar
        subplot(2,3,2);imagesc(texture2dx1dy0); title('GLCM: 0   degree ');colorbar
        subplot(2,3,3);imagesc(texture2dx1dymin1);title('GLCM: 45  degree ');colorbar
        subplot(2,3,4);imagesc(texture2dx0dymin1);title('GLCM: 90  degree ');colorbar
        subplot(2,3,5);imagesc(texture2dxmin1dymin1);title('GLCM: 135 degree ');colorbar
        subplot(2,3,6);imagesc((texture2dx1dy0+texture2dx0dymin1)*0.5);title('GLCM: 0 + 90 degree ');colorbar
        
    elseif(idata==3)
        
        figure;
        suptitle(sprintf('=== Texture: %d ===\n',idata));
        subplot(2,3,1);imagesc(mosaic1_train(257:512,1:256));title('Input Image');colorbar
        subplot(2,3,2);imagesc(texture3dx1dy0); title('GLCM: 0   degree ');colorbar
        subplot(2,3,3);imagesc(texture3dx1dymin1);title('GLCM: 45  degree ');colorbar
        subplot(2,3,4);imagesc(texture3dx0dymin1);title('GLCM: 90  degree ');colorbar
        subplot(2,3,5);imagesc(texture3dxmin1dymin1);title('GLCM: 135 degree ');colorbar
        subplot(2,3,6);imagesc((texture3dx1dy0+texture3dx0dymin1)*0.5);title('GLCM: 0 + 90 degree ');colorbar
        
    else
        figure;
        suptitle(sprintf('=== Texture: %d ===\n',idata));
        subplot(2,3,1);imagesc(mosaic1_train(257:512,257:512));title('Input Image');colorbar
        subplot(2,3,2);imagesc(texture4dx1dy0); title('GLCM: 0   degree ');colorbar
        subplot(2,3,3);imagesc(texture4dx1dymin1);title('GLCM: 45  degree ');colorbar
        subplot(2,3,4);imagesc(texture4dx0dymin1);title('GLCM: 90  degree ');colorbar
        subplot(2,3,5);imagesc(texture4dxmin1dymin1);title('GLCM: 135 degree ');colorbar
        subplot(2,3,6);imagesc((texture4dx1dy0+texture4dx0dymin1)*0.5);title('GLCM: 0 + 90 degree ');colorbar
    end

end

%% Part-2: Discussing new features by subdividing the GLCM matrices

%define the selected GLCM
texture1_glcm = (texture1dx1dy0+texture1dx0dymin1)*0.5;
texture2_glcm = (texture2dx1dy0+texture2dx0dymin1)*0.5;
texture3_glcm = (texture3dx1dy0+texture3dx0dymin1)*0.5;
texture4_glcm = (texture4dx1dy0+texture4dx0dymin1)*0.5;


%derive the Q1, Q2, Q3 and Q4 features based on selected GLCM
texture1_Feature_Qs = nan(1,8);
texture2_Feature_Qs = nan(1,8);
texture3_Feature_Qs = nan(1,8);
texture4_Feature_Qs = nan(1,8);

for idata = 1:4
    if(idata==1)
        texture1_Feature_Qs(1) = sum(sum(texture1_glcm(1:8,1:8)))/sum(texture1_glcm(:));
        texture1_Feature_Qs(2) = sum(sum(texture1_glcm(1:8,9:16)))/sum(texture1_glcm(:));
        texture1_Feature_Qs(3) = sum(sum(texture1_glcm(9:16,1:8)))/sum(texture1_glcm(:));
        texture1_Feature_Qs(4) = sum(sum(texture1_glcm(9:16,9:16)))/sum(texture1_glcm(:));
        texture1_Feature_Qs(5) = sum(sum(texture1_glcm(1:4,1:4)))/sum(texture1_glcm(:));
        texture1_Feature_Qs(6) = sum(sum(texture1_glcm(1:4,5:8)))/sum(texture1_glcm(:));
        texture1_Feature_Qs(7) = sum(sum(texture1_glcm(5:8,1:4)))/sum(texture1_glcm(:));
        texture1_Feature_Qs(8) = sum(sum(texture1_glcm(5:8,5:8)))/sum(texture1_glcm(:));       

    elseif(idata==2)
        texture2_Feature_Qs(1) = sum(sum(texture2_glcm(1:8,1:8)))/sum(texture2_glcm(:));
        texture2_Feature_Qs(2) = sum(sum(texture2_glcm(1:8,9:16)))/sum(texture2_glcm(:));
        texture2_Feature_Qs(3) = sum(sum(texture2_glcm(9:16,1:8)))/sum(texture2_glcm(:));
        texture2_Feature_Qs(4) = sum(sum(texture2_glcm(9:16,9:16)))/sum(texture2_glcm(:));
        texture2_Feature_Qs(5) = sum(sum(texture2_glcm(1:4,1:4)))/sum(texture2_glcm(:));
        texture2_Feature_Qs(6) = sum(sum(texture2_glcm(1:4,5:8)))/sum(texture2_glcm(:));
        texture2_Feature_Qs(7) = sum(sum(texture2_glcm(5:8,1:4)))/sum(texture2_glcm(:));
        texture2_Feature_Qs(8) = sum(sum(texture2_glcm(5:8,5:8)))/sum(texture2_glcm(:));
        
    elseif(idata==3)
        texture3_Feature_Qs(1) = sum(sum(texture3_glcm(1:8,1:8)))/sum(texture3_glcm(:));
        texture3_Feature_Qs(2) = sum(sum(texture3_glcm(1:8,9:16)))/sum(texture3_glcm(:));
        texture3_Feature_Qs(3) = sum(sum(texture3_glcm(9:16,1:8)))/sum(texture3_glcm(:));
        texture3_Feature_Qs(4) = sum(sum(texture3_glcm(9:16,9:16)))/sum(texture3_glcm(:));
        texture3_Feature_Qs(5) = sum(sum(texture3_glcm(1:4,1:4)))/sum(texture3_glcm(:));
        texture3_Feature_Qs(6) = sum(sum(texture3_glcm(1:4,5:8)))/sum(texture3_glcm(:));
        texture3_Feature_Qs(7) = sum(sum(texture3_glcm(5:8,1:4)))/sum(texture3_glcm(:));
        texture3_Feature_Qs(8) = sum(sum(texture3_glcm(5:8,5:8)))/sum(texture3_glcm(:));
        
    else
        texture4_Feature_Qs(1) = sum(sum(texture4_glcm(1:8,1:8)))/sum(texture4_glcm(:));
        texture4_Feature_Qs(2) = sum(sum(texture4_glcm(1:8,9:16)))/sum(texture4_glcm(:));
        texture4_Feature_Qs(3) = sum(sum(texture4_glcm(9:16,1:8)))/sum(texture4_glcm(:));
        texture4_Feature_Qs(4) = sum(sum(texture4_glcm(9:16,9:16)))/sum(texture4_glcm(:));
        texture4_Feature_Qs(5) = sum(sum(texture4_glcm(1:4,1:4)))/sum(texture4_glcm(:));
        texture4_Feature_Qs(6) = sum(sum(texture4_glcm(1:4,5:8)))/sum(texture4_glcm(:));
        texture4_Feature_Qs(7) = sum(sum(texture4_glcm(5:8,1:4)))/sum(texture4_glcm(:));
        texture4_Feature_Qs(8) = sum(sum(texture4_glcm(5:8,5:8)))/sum(texture4_glcm(:));
    end 
    
end

%plot the texture features
figure;
suptitle(sprintf('=== Comparison of texture features === \n\n'))
subplot(2,2,1);bar(texture1_Feature_Qs);ylim([0,1]);title('Features of texture1');xlabel('Features Type');ylabel('Feature Value');grid on;
subplot(2,2,2);bar(texture2_Feature_Qs);ylim([0,1]);title('Features of texture2');xlabel('Features Type');ylabel('Feature Value');grid on;
subplot(2,2,3);bar(texture3_Feature_Qs);ylim([0,1]);title('Features of texture3');xlabel('Features Type');ylabel('Feature Value');grid on;
subplot(2,2,4);bar(texture4_Feature_Qs);ylim([0,1]);title('Features of texture4');xlabel('Features Type');ylabel('Feature Value');grid on;


%% Part-3: Selecting and implementing a subset of these features.
%convert the image from the original 8 bit(2^8 =256 gray level) to 4 bit (2^4 =16 gray level) and equalize the mosaic image
gray_level        = 16;
mosaic1_traindata = uint8(round(double(mosaic1_train) * (gray_level-1) / double(max(mosaic1_train(:)))));
mosaic2_testdata  = uint8(round(double(mosaic2_test) * (gray_level-1) / double(max(mosaic2_test(:)))));
mosaic3_testdata  = uint8(round(double(mosaic3_test) * (gray_level-1) / double(max(mosaic2_test(:)))));


%set window size for sliding window GLCM analysis:
window     = [31,31];

%save the derived features for classification use
GLCM_features = cell(3,8);

for n = 1:3
    %define the GLCM analyzing parameters (step and angle)
    %-- use 0 and 90 degree GLCM --
    dx = [1,0]; 
    dy = [0,-1];
    
    switch n
        case 1 % isotropic GLCM
            imag = mosaic1_traindata;
        case 2 % 90 degree GLCM
            imag = mosaic2_testdata;
        otherwise
            imag = mosaic3_testdata;
    end
    
    %start sliding window GLCM calculation
    [Quadrant_features] = Sliding_Window_GLCM_Features(imag,gray_level,dx,dy,window);
    
    % display the derived GLCM feature matrix
    figure;
    subplot(2,3,1)
    imagesc(imag);
    colorbar;
    title(sprintf('input Image: %d',n));
    subplot(2,3,2)
    imagesc(Quadrant_features{4});
    colorbar;
    title('Quadrant features-4')
    subplot(2,3,3)
    imagesc(Quadrant_features{5});
    colorbar;
    title('Quadrant features-1-sub-1')
    subplot(2,3,4)
    imagesc(Quadrant_features{6});
    colorbar;
    title('Quadrant features-1-sub-2')
    subplot(2,3,5)
    imagesc(Quadrant_features{7});
    colorbar;
    title('Quadrant features-1-sub-3')
    subplot(2,3,6)
    imagesc(Quadrant_features{8});
    colorbar;
    title('Quadrant features-1-sub-4')
    
    %output the GLCM quarantic features
    GLCM_features{n,1} = Quadrant_features{1};
    GLCM_features{n,2} = Quadrant_features{2};
    GLCM_features{n,3} = Quadrant_features{3};
    GLCM_features{n,4} = Quadrant_features{4};
    GLCM_features{n,5} = Quadrant_features{5};
    GLCM_features{n,6} = Quadrant_features{6};
    GLCM_features{n,7} = Quadrant_features{7};
    GLCM_features{n,8} = Quadrant_features{8};
end


%% Part-4: Implement a multivariate Gaussian classifier.

%---- check the enclosed function 'Multivariate_Gaussian_Classifier.m' for multivariate gaussian classifier 

%% Part-5: Training the classifier based on the feature subset from point 3

%load the calculated GLCM features for training dataset and testing datasets
test_input    = load('output_project2.mat');
GLCM_features = test_input.GLCM_features;
ClassLabels   = test_input.training_mask;

%sort the selected features from training datasets into 3d matrix
TainingFeaturesArray        = zeros(512,512,3);
TainingFeaturesArray(:,:,1) = GLCM_features{1,5};
TainingFeaturesArray(:,:,2) = GLCM_features{1,6};
TainingFeaturesArray(:,:,3) = GLCM_features{1,8};

%apply classification based on multivariate Gaussian classifier.
[ class_predict_trainingset ] = Multivariate_Gaussian_Classifier(TainingFeaturesArray,ClassLabels,TainingFeaturesArray);

%plot the confusion matrix
% ConfusionMat = confusionmat(double(ClassLabels(:)),class_predict_trainingset(:)); 
% plotconfusion(double(ClassLabels(:)),class_predict_trainingset(:));

%% Part-6:  Evaluation of classification performance on the test dataset using the set of features selected in point 3
%sort the selected features from testing datasets into 3d matrix
Test1FeaturesArray          = zeros(512,512,3);
Test1FeaturesArray(:,:,1)   = GLCM_features{2,5};
Test1FeaturesArray(:,:,2)   = GLCM_features{2,6};
Test1FeaturesArray(:,:,3)   = GLCM_features{2,8};

Test2FeaturesArray          = zeros(512,512,3);
Test2FeaturesArray(:,:,1)   = GLCM_features{3,5};
Test2FeaturesArray(:,:,2)   = GLCM_features{3,6};
Test2FeaturesArray(:,:,3)   = GLCM_features{3,8};

%apply classification based on multivariate Gaussian classifier.
[ class_predict_test1 ]       = Multivariate_Gaussian_Classifier(TainingFeaturesArray,ClassLabels,Test1FeaturesArray);
[ class_predict_test2 ]       = Multivariate_Gaussian_Classifier(TainingFeaturesArray,ClassLabels,Test2FeaturesArray);


