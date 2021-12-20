n=10413
h = 2/(n+1)

for  i= 1: n
 x= -1 + 2*i/(n+1);
 if x<=0
     f(i)=x+1;
 end
 if x>0
     f(i)=-x+1;
 end
 f(i)=f(i)*h^2;
end

for i=1:n
    for j=1:n
        if i-1==j 
            A(i,j)=1;
            
        end
        if i==j
            A(i,j)=-2;
            
        end
        if i+1==j
            A(i,j)=1;
            
        end
    end
end
Forbenisova=norm(A,'fro')
DETERMINANTA=det(A)
u=linsolve(A,f.')
PRVA_NORMA=norm(u,1)