param n integer >0; #Quantité de clients
param M := 1000; #Numero grande
param W>0;
param P {i in 0..n};
param CX {i in 0..n};
param CY {i in 0..n};

param D {i in 0..n, j in 0..n}:= if i=j then 1000 else sqrt( (CX[j]-CX[i])^2 +(CY[j]-CY[i])^2 );

var X{i in 0..n, j in 0..n} binary;
var t{i in 0..n} >=0 <=W;

maximize Z: sum{i in 0..n, j in 1..n: i!=j} (P[j]*X[i,j]);

visit{j in 0..n}: sum{i in 0..n: i!=j} X[i,j] <= 1;

sale{j in 0..n}: sum{i in 0..n} X[i,j] = sum{i in 0..n} X[j,i];

inicio{i in 1..n}: t[i]>=D[0,i]*X[0,i];

subtour{i in 1..n, j in 0..n: i!=j}: t[i] + D[i,j] <= t[j] + M*(1- X[i,j]);

solve;
display Z;

display {i in 0..n, j in 0..n : X[i,j]=1} X[i,j];

display {i in 0..n : sum{j in 0..n}X[j,i]!=0} t[i];

display {i in 0..n, j in 0..n : X[i,j]=1} D[i,j];

data;
param n:=10;
param W:=100;
param:	CX CY P :=
0	6	45	0
1	14	2	44
2	44	37	35
3	22	45	47
4	43	8	45
5	7	26	11
6	24	16	23
7	20	13	8
8	47	19	17
9	35	28	9
10	43	17	49
;
end;
