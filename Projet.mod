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
	0	20	44	0
	1	20	20	15
	2	22	19	12
	3	41	2	32
	4	18	40	37
	5	7	7	30
	6	12	10	36
	7	1	48	2
	8	24	28	17
	9	14	34	44
	10	21	8	11
;
end;
