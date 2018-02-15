clear all; close all;

I = imread('Edison.jpg');
I = imresize(I,0.5);

% Expand image to visualization purposes
I = [zeros(size(I,1),150,3),I,zeros(size(I,1),300,3)];

% Inline function for drawing a line
x = [1,size(I,2)]; % x-coordinates
y = @(l) (-l(3)-l(1)*x)/l(2); % y-coordinates

% Homogeneous corner points of the building
p1 = [196 220 1];
p2 = [321 137 1];
p3 = [197 314 1];
p4 = [321 303 1];
p5 = [384 114 1];
p6 = [501 181 1];
p7 = [384 305 1];
p8 = [501 311 1];
  
figure(1)
imshow(I);
hold on;
plot(p1(1),p1(2),'r*',p2(1),p2(2),'r*',p3(1),p3(2),'b*',p4(1),p4(2),'b*',...
    p5(1),p5(2),'g*',p6(1),p6(2),'g*',p7(1),p7(2),'c*',p8(1),p8(2),'c*')

% Labels corresponding to the 8 corner points, p1-p8
labels = cellstr( num2str([1:8]') );
points = [p1;p2;p3;p4;p5;p6;p7;p8];
text(points(:,1), points(:,2), labels, 'VerticalAlignment','bottom', ...
                             'HorizontalAlignment','right', 'FontSize', 14)
                         
% Step 1: compute the lines l1-l4 between the pairwise points.
% l1=[a,b,c] is the line between p1 and p2 satisfying the equation
% ax+by+c=0
% To plot a line, l1, use: plot(x,y(l1))



% Step 2: compute the first vanishing points, pv1, as the intersection
% between the two lines, l1 and l2
% To plot a point, p1, use: plot(p1(1),p1(2),'x')



% Step 3: repete step 1 and 2 for points p5-p8



% Step 4: compute and visualize the horizon


% Exercise: compute the height of the camera above the ground
