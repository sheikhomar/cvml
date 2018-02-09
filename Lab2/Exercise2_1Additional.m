clear all; close all;

boxImage = im2single(rgb2gray(imread('corners.png')));
figure;
imshow(boxImage);
title('Image of a Box');

boxPoints = detectSURFFeatures(boxImage);
figure;
imshow(boxImage);
title('100 Strongest Feature Points from Box Image');
hold on;
plot(selectStrongest(boxPoints, 100));

points = detectHarrisFeatures(boxImage);
figure;
imshow(boxImage);
title('100 Strongest Feature Points from Box Image');
hold on;
plot(selectStrongest(points, 100));