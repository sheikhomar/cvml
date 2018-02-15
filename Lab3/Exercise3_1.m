% Exercise 1
clear;  clc;
% Define a 2-by-2 square centered at (xc,yc) = (4,4). Note that the
% position of the upper right corner of the square is dublicated to
% ensure that the square is closed when plotting it.
xc = 0;
yc = 0;
square = [ xc+1 xc+1 xc-1 xc-1 xc+1;
           yc+1 yc-1 yc-1 yc+1 yc+1];
figure(1)
plot(square(1,:),square(2,:),'-',xc,yc,'.')
legend('Square','Center of square','Location','SouthEast')
xlabel('X')
ylabel('Y')
axis([-2 6 -2 6])
axis equal

% Convert square to homogeneous coordinates
square_homo = [ square;
                ones(1,5) ];

% Step 1:
% Define a 3-by-3 transformation matrix (A) that rotates the square by 45
% degrees (counter-clockwise) around its center point (xc,yc).
         
A = ???




square_homo_rotated = A*square_homo;
figure(1)
hold on
plot(square_homo_rotated(1,:),square_homo_rotated(2,:),'r-')
hold off
legend('Square','Center of square','Square rotated 45 degrees','Location','SouthEast')

%% Transformations in 3D

% xc = 4;
% yc = 4;
% zc = 3;
% cube = [ xc+1 xc+1 xc-1 xc-1 xc+1 xc+1 xc+1 xc-1 xc-1 xc+1 xc+1 xc+1 xc-1 xc-1 xc-1 xc-1;
%            yc+1 yc-1 yc-1 yc+1 yc+1 yc+1 yc-1 yc-1 yc+1 yc+1 yc-1 yc-1 yc-1 yc-1 yc+1 yc+1;
%            zc-1 zc-1 zc-1 zc-1 zc-1 zc+1 zc+1 zc+1 zc+1 zc+1 zc+1 zc-1 zc-1 zc+1 zc+1 zc-1];
% fig = figure()
% plot3(cube(1,:),cube(2,:),cube(3,:),'-',xc,yc,zc,'.')
% legend('Cube','Center of cube','Location','SouthEast')
% xlabel('X')
% ylabel('Y')
% zlabel('Z')
% axis equal
% axis([-2 6 -2 6 -2 6])
% grid on
% 
% % Convert cube to homogeneous coordinates
% cube_homo = [ cube;
%                 ones(1,size(cube,2)) ];
%              
% % A = ???
% 
% cube_homo_rotated = A*cube_homo;
% figure(fig)
% hold on
% plot3(cube_homo_rotated(1,:),cube_homo_rotated(2,:),cube_homo_rotated(3,:),'r-')
% hold off
% legend('Cube','Center of cube','Cube rotated 45 degrees','Location','SouthEast')
