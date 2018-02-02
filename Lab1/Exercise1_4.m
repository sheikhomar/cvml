% Task1

close all;

% Apply histogram equalization on each of the three color channels of the
% image
I = imread( 'fruits.jpg' );
figure;
imshow(I);

R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);
    
R = histeq(R);
G = histeq(G);
B = histeq(B);

equalized = cat( 3, R, G, B );

figure;
imshow(equalized);

% Apply histogram equalization on the intensity channel of the image when
% represented in the HSV color space

I = imread( 'fruits.jpg' );
figure;
imshow(I);

% here transform the image to HSV and apply histogram equalization
% you can use the matlab functions rgb2hsv() and hsv2rgb()
Ieq = intensityeq(I);  

figure;
imshow(Ieq);

I = imread( 'festia.jpg' );
figure;
imshow(I);

Ieq = intensityeq(I);

figure;
imshow(Ieq);

