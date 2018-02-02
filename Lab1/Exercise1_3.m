close all;  clear all;

[I,map]=imread('spine.jpg'); 
figure(1), imshow(I);

% calculate image histogram
level = 0:255;
vI = double(I(:));
for i=0:255
    histI(i+1) = ???
end
histI = histI/sum(histI); %Normalize w/total pixels
figure(2), plot(histI);

% cumulative distrib
for i = 1:length(level)
    chistI(i) = sum(histI(1:i));
end
figure(3)
plot(level,chistI)

minc = min(chistI);
%f = round((chistI-minc)/(1-minc)*255+0.5);
f = round(chistI*255+0.5);
eqI = f(double(I)+1);

figure(4)
imshow(uint8(eqI),gray(256));
figure(5)
imhist(uint8(eqI),gray(256));


ContrastStretch(I,0,255);
