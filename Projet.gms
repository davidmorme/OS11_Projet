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
1        26
2        41
3        4
4        22
5        48
6        20
7        22
8        21
9        28
10        4
11        42
/

cy(i)
/
1        46
2        33
3        38
4        41
5        32
6        22
7        10
8        27
9        48
10        1
11        41
/
P(i)
/
1        0
2        11
3        29
4        46
5        28
6        11
7        5
8        34
9        38
10        41
11        13
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
