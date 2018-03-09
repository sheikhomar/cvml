clear all;close all;

% Set up some paths
VLFEATROOT = 'vlfeat-0.9.20';
%run([VLFEATROOT '\toolbox\vl_setup.m']);
addpath([VLFEATROOT,'\toolbox'])
vl_setup()

addpath('MatlabSourceFiles');

% Load test image:
I = imread('bag.png');
I=double(I)/255;
figure,imshow(I);

% Filter input image with 4 directional filters. Each filter is the
% spatial derivative of a 2D Gaussian.
angles = (0:45:135)/180*pi;
sigma = 5;
display = 1;
disp('Press any key to continue')
H=steerGaussMultipleAngles(I,angles,sigma,display);

numClust = 4;

% Exercise: create 4-dimensional feature vectors and cluster them into 'numClust'
% clusters. 'H' contains the responses of the 4 directional filters. Use
% '[C,Ix] = vl_kmeans(datapoints,numClust)' to perform the clustering.
% 'C' denotes the 4 cluster centers. A is a row vector specifying the 
% assignments of 'datapoints' to these 'numClust' centers.

% datapoints = ???
% [C,Ix] = vl_kmeans(datapoints,numClust)


% Exercise: construct and display the texton map. Use the assignments of 
% pixels to the clusters from kmeans to construct 'textonMap'. Each pixel
% should be assigned a cluster number as its intensity.
% Hint: use show_norm_image(img) for visualization.

% textonMap = ???
% show_norm_image(textonMap)

% Show texton histograms of top half and bottom half of image
tophalf_image = I(1:125,:);
bottomhalf_image = I(126:250,:);
tophalf_textonmap = textonMap(1:125,:);
bottomhalf_textonmap = textonMap(126:250,:);
figure
subplot(3,2,1),imshow(tophalf_image),title('Top half of image')
subplot(3,2,2),imshow(bottomhalf_image),title('Bottom half of image')
subplot(3,2,3),show_norm_image(tophalf_textonmap),title('Top half texton map (=texton 1)')
subplot(3,2,4),show_norm_image(bottomhalf_textonmap),title('Bottom half texton map (=texton 2)')
subplot(3,2,5),hist(tophalf_textonmap(:),1:numClust),title('Histogram of texton 1')
subplot(3,2,6),hist(bottomhalf_textonmap(:),1:numClust),title('Histogram of texton 2')

% See if we can detect the horizontal edge between texture 1 and 2 using a
% half-disc like filter.
filter_width = 40;
half_filter_width = filter_width/2;
difference = zeros(1,250);
for row = half_filter_width+1 : 250-half_filter_width
    half1 = textonMap(row-half_filter_width:row-1,:);
    half2 = textonMap(row:row+half_filter_width,:);
    g = hist(half1(:),1:numClust);
    h = hist(half2(:),1:numClust);
    difference(row) = 0.5*sum((g-h).^2./(g+h));
end

% Find the largest edge (the maximum difference of histograms)
[maxval,maxix] = max(difference);

figure
subplot(1,2,1),plot(1:250,difference,maxix,maxval,'r*')
legend('Difference','Maximum difference','Location','NorthOutside') 
title('Difference between histograms');
xlabel('Image row coordinate')
ylabel('Difference')
subplot(1,2,2),imshow(I)
line([1 size(I,1)],[maxix maxix])
title('Detected edge')










