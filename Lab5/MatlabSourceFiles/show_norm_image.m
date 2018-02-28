function show_norm_image(I)

I=I-min(I(:));
I=I/max(I(:));
imshow(I);