function [x0,y0,r]=threepoint(p1,p2,p3)

x1=p1(1);
x2=p2(1);
x3=p3(1);

y1=p1(2);
y2=p2(2);
y3=p3(2);

a=det([x1 y1 1;x2 y2 1;x3 y3 1]);
d=-det([x1*x1+y1*y1 y1 1;x2*x2+y2*y2 y2 1;x3*x3+y3*y3 y3 1]);
e=det([x1*x1+y1*y1 x1 1;x2*x2+y2*y2 x2 1;x3*x3+y3*y3 x3 1]);
f=-det([x1*x1+y1*y1 x1 y1;x2*x2+y2*y2 x2 y2;x3*x3+y3*y3 x3 y3]);

x0=-d/(2*a);
y0=-e/(2*a);
r=sqrt((d*d+e*e)/(4*a*a)-f/a);