S=0.5
rho=1.3
c=1
m=80
g=9.8
tmax1=287/100
tmax2=2*tmax1
h1=(tmax1-0)/50
h2=(tmax2-0)/50
x0=1500
v0=0
t0=0
B=1/2*c*rho*S/m
fv=@(v,t) -g+B*v^2
fx=@(v,t) v


%x - xdata, y - ydata
%returns interpolation polynomial
U=RK4(fx, fv, t0,x0, v0, tmax1, h1)
V=RK4(fx,fv,t0,x0,v0,tmax2,h2)


function u = RK4_step(f, x, y, h)
%x - xdata, y - ydata
%returns interpolation polynomial
    k1 = h * f(x,y);
    k2 = h * f(x + 0.5*h, y + 0.5*k1); 
    k3 = h * f(x + 0.5*h, y + 0.5*k2); 
    k4 = h * f(x + h, y + k3);
    
    u = y + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6;
        
end

function u = RK4(fx, fv, t0, x0, v0, tmax, h)
    %x - xdata, y - ydata
    %returns interpolation polynomial
    t = [t0 : h : tmax]';
    x = [x0];
    v = [v0];
    for i = t0:h:tmax -h
        x = [x; RK4_step(fx, v(length(v)), x(length(x)), h)];
        v = [v; RK4_step(fv, i, v(length(v)), h)];    
    end
    u = [t x v];            
end

