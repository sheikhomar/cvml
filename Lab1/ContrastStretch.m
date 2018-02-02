function II = ContrastStretch(im, lo, hi)
im = double(im);
immax = max(max(im));
immin = min(min(im));
im1 = (im - immin)*(hi-lo)/(immax-immin)+lo;
figure, imshow(uint8([im im1])); title('Stretch the original image (left) to a new range on the right.');

end