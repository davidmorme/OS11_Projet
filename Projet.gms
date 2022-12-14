SETS
i /1*11/
;alias(i,j)
scalar
W /100/
;
parameters
D(i,j)
cx(i)
/
1        6
2        14
3        44
4        22
5        43
6        7
7        24
8        20
9        47
10        35
11        43
/

cy(i)
/
1        45
2        2
3        37
4        45
5        8
6        26
7        16
8        13
9        19
10        28
11        17
/
P(i)
/
1        0
2        44
3        35
4        47
5        45
6        11
7        23
8        8
9        17
10        9
11        49
/

;
D(i,j) = sqrt(power(cx(i)-cx(j),2)+power(cy(i)-cy(j),2))   ;

binary variable
X(i,j)
;
Free variable
Z
;
Positive variable
t(i)
;
Equations
FO
R1
R2
R3
R4
R5
;


FO..     Z=E=sum((i,j)$(ord(i)<>ord(j) and ord(j)>1),P(j)*X(i,j));
R1(j)..  sum(i $(ord(i)<>ord(j)), X(i,j)) =L= 1;
R2(j)..  sum(i $(ord(i)<>ord(j)), X(i,j)) =E= sum(i $(ord(i)<>ord(j)), X(j,i));
R3(i)$(ord(i)>1)..  t(i)=G=D('1',i)*X('1',i);
R4(i,j) $ ((ord(i)<>ord(j))and ord(i)>1 ).. t(i) + D(i,j) =L= t(j) + 10000*(1- X(i,j));
R5(i)..  t(i)=L=W;

MODEL problem /ALL/;
solve problem using mip maximizing Z;
Option  Optca=5;
option Optcr=0.1;
Option Reslim=10000;
Option Seed=3000;
display Z.l, X.l, t.l;
