close all 

% Read mask
class_train_mask = double(imread('tm_train.png'));

num_classes = 4;

% Estimate a priory probability (if one want to)
P = zeros(1,num_classes);
for ii = 1:num_classes 
    P(ii) = sum(sum(class_train_mask == ii)); 
end
P = P./sum(P);

% Read in the images
n = 260; m = 333;
num_bands = 6;

data = zeros(n,m,num_bands);
for ii = 1:num_bands
    data(:,:,ii) = imread(sprintf('tm%d.png',ii));
end

% Classify
est_imgs = zeros(n,m,num_bands);

for ii = 1:num_bands
    tmp_data = double(data(:,:,ii));
    
    class_means = zeros(1,num_classes);
    class_stds = zeros(1,num_classes);
    
    % train
    for jj = 1:num_classes 
        class_tmp_data = tmp_data(class_train_mask == jj); 
        
        class_means(jj) = mean(class_tmp_data);
        class_stds(jj) = sqrt(var(class_tmp_data)); 
    end
    
    % perform the classification 
    for k = 1:n
        for l = 1:m
            
            % outcomment P in the calculation for p if you wish to assume
            % uniform a priory probability for each class, as it won't
            % affect the value of p for each class.
            
            p = (1./class_stds) .* exp(-.5.*( (tmp_data(k,l) - class_means) ./ class_stds ).^2 );%.*P; % factor 1/sqrt(2*pi) does not have that much to say. 
            [~, est_imgs(k,l,ii)] = max(p);
        end
    end
    
end

% Find test accuracy 

class_test_mask = imread('tm_test.png');
tot_test = sum(sum(class_test_mask > 0));

for ii = 1:num_bands
    test_data = data(:,:,ii);
    estimated_data = est_imgs(:,:,ii);
    
    tot_correct = 0;
    fprintf('band %d -----\n',ii);
    for jj = 1:num_classes
        class_est_test = estimated_data(class_test_mask == jj);
        num_correct = sum(class_est_test == jj);
        tot_correct = tot_correct + num_correct;
        fprintf('for class %d, there was %g percent correctly classifications.\n', ...
                jj, num_correct/numel(class_est_test == jj)*100);
    end
    
    fprintf('Total number of correct classifications: %g percent\n',tot_correct/tot_test*100);
end

% Plot original images and classified images

figure()
for ii = 1:num_bands
    subplot(2,3,ii)
    imagesc(data(:,:,ii))
end

figure()
for ii = 1:num_bands
    subplot(2,3,ii)  
    imagesc(est_imgs(:,:,ii))
    title(sprintf('%d',ii))
    colormap(parula(num_classes));
    colorbar('Ticks',1:4 );
end

figure()
imagesc(class_test_mask)