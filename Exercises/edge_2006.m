% Hough accumulation

clear all
close all

% Start by making image

im=zeros(16);
im(1,1)=1;
im(5,5)=1;
im(8,8)=1;
im(1,16)=1;
im(16,1) = 1;
figure
imshow(im)

% The question is, are these point on a line?
% That is, is there a line y=ax+b that passes
% through these points for some values of
% a and b?

% Take every x,y coordinate from the given image and plug it into
% the equation b=-xa+y AS PARAMETERS. Then vary a over some
% predefined interval and take a look at the lines produced by this:
% y = ax + b

figure

% Point 1, x=1, y=1, b=-a+1

a=-5:5
b=-a+1
plot(a,b,'r')
hold on

% Point 2, x=5, y=5, b=-5a+5

a=-5:5
b=-5*a+5
plot(a,b,'g')

% Point 3, x=9, y=9, b=-9a+9

a=-5:5
b=-8*a+8
plot(a,b,'b')

% Point 4, x = 1, y = 16; b = -a + 16
a=-5:5
b=-a+16
plot(a,b,'k');

% Point 5 x = 16, y = 1; b = -16a + 1
a=-5:5
b=-16*a+1
plot(a,b,'y');

grid on
axis([-10 10 -20 20])
xlabel('a')
ylabel('b')



%% What could the accumulator matrix look like
acc=zeros(11,176);
figure

a=-5:5
b=-a+1
for x=1:length(a)
    acc(a(x)+6,b(x)+80)=acc(a(x)+6,b(x)+80)+1;
end
imshow(acc,[])
xlabel('b'),ylabel('a')

a=-5:5
b=-5*a+5
for x=1:length(a)
    
    acc(a(x)+6,b(x)+80)=acc(a(x)+6,b(x)+80)+1;
end
imshow(acc,[])
xlabel('b'),ylabel('a')

a=-5:5
b=-9*a+9
for x=1:length(a)
    acc(a(x)+6,b(x)+80)=acc(a(x)+6,b(x)+80)+1;
end
imshow(acc,[])
xlabel('b'),ylabel('a')

figure
mesh(acc)

%% Hough transform example 1

clear all
close all

i=zeros(11);
for j=3:9
  i(j,j)=1;
  i(j+3,j) = 1;
end
  
figure
colormap(gray(2))
imagesc(i);
grid on
set(gca,'xcolor','w')
set(gca,'ycolor','w')
axis image

theta=0:179;
[iH,Xp]=houghradon(i,theta);
figure
imagesc(theta,Xp,iH)
colormap(cool)
colorbar
xlabel('\theta (degrees)')
ylabel('\rho (pixels from center)')
grid on

%% Hough transform example 2

clear all
close all

i=zeros(11);
for j=3:9
  i(j,j)=1;
end

i(5,5)=0;
i(7,7)=0;
  
figure
colormap(gray(2))
imagesc(i);
grid on
set(gca,'xcolor','w')
set(gca,'ycolor','w')
axis image

theta=0:179;
[iH,Xp]=houghradon(i,theta);
figure
imagesc(theta,Xp,iH)
colormap(cool)
colorbar
xlabel('\theta (degrees)')
ylabel('\rho (pixels from center)')
grid on

%% Hough transform example 3


clear all
close all

i=imread('corridor.png');
ig=double(rgb2gray(i));

figure
imshow(ig,[min(min(ig)) max(max(ig))])

h1=fspecial('sobel');
h2=h1';
igh=imfilter(ig,h1);
igv=imfilter(ig,h2);
igs=abs(igh)+abs(igv);
figure
imshow(igs,[min(min(igs)) max(max(igs))])

igsT=igs>170;
figure
imshow(igsT)

theta=0:179;
[igsH,Xp]=houghradon(igsT,theta);
figure
imagesc(theta,Xp,igsH)
colormap(hot)
colorbar
xlabel('\theta (degrees)')
ylabel('\rho (pixels from center)')
grid on

ind=find(igsH>100);
[dummy,index]=sort(-igsH(ind));
k=ind(index(1:20))
[r,c]=ind2sub(size(igsH),k);

t=-theta(c)*pi/180;
rho=Xp(r);

lines = [cos(t)' sin(t)' -rho];  
cx = size(ig,2)/2-1;
cy = size(ig,1)/2-1;
lines(:,3) = lines(:,3) - lines(:,1)*cx - lines(:,2)*cy;  

figure
imshow(igs,[min(min(igs)) max(max(igs))])
draw_lines(lines);

figure
s=surf(igsH)
set(s,'edgecolor','none')
shading interp
lighting phong
camlight right
set(gcf,'color','k')
set(gca,'color','k')