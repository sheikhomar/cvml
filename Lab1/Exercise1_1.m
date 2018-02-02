clear all
close all
clc
 
I = double(imread('lena.jpg'))./255;
 
M=0; N=0.005;
I_gaus = imnoise(I,'gaussian',M,N);
 
D=0.02;
I_salt = imnoise(I,'salt & pepper',D);
 
a=0.2;b=0.02;[M N]= size(I);
Ray_noise = a + sqrt(-b*log(1-rand(M,N)));
I_ray = I + Ray_noise;
 
colormap('gray');
subplot(2,2,1);imagesc(I);      title('Original');
subplot(2,2,2);imagesc(I_gaus); title('Gaussian');
subplot(2,2,3);imagesc(I_salt); title('Salt & Pepper');
subplot(2,2,4);imagesc(I_ray);  title('Rayleigh');
 
m=5;n=5;
images = cell(1,3);
images{1} = I_gaus; images{2} = I_salt; images{3} = I_ray;
 
for i=1:3
    g=images{i};
    % Arithmetic Mean
    f1 = ???
 
    % Geometric Mean
    f2 = ???
 
    % Harmonic Mean
    f3 = ???
 
    figure; colormap('gray');
    subplot(2,2,1);imagesc(g);  title('Original');
    subplot(2,2,2);imagesc(f1); title('Arithmetic Mean');
    subplot(2,2,3);imagesc(f2); title('Geometric Mean');
    subplot(2,2,4);imagesc(f3); title('Harmonic Mean');
end
