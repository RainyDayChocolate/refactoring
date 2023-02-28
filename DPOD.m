clear all
clc
close all
  load 4.mat
 sizeofD=100;
 dataset2=dataset4(:,1:3);
S=histogram(dataset2,sizeofD);
S=S.Values;
eps=linspace(0.1,2,4);
B=S';
C=B(1:sizeofD);
[Output]=kse_test(C);
pp=1./(Output.^5);
pp=pp';
NN=2.^(15*Output);
px=[1:1:sizeofD];
warning('off','all')

gamma=0.5;
rho = exp(lambertw(-1,-gamma / (2 * exp(0.5))) + 0.5);
m = log(1/rho)/(2*(gamma-rho)^2);
	k = (m * (1 - gamma + rho + sqrt(log(1/rho)/(2 * m))));
	m=ceil(m);
    k=ceil(k);

EVAL=zeros(4,7);
EVAL1=zeros(4,7);
for rr=1:4


for j=1:1
    C=B(1:sizeofD);
GS=datasample(C,m);
GS=sort(GS);
Sensunif=GS(k)
SS=randpdf(pp,px,[m,1]);
SS=NN(floor(SS));
SS=sort(SS);
Sensanom=SS(k)

unif=laprnd(sizeofD, 1, 0,Sensunif/(eps(rr)))+C;
noiseanom=laprnd(sizeofD, 1, 0,Sensanom/(eps(rr)))+NN;


[noiseunif]=kse_test(unif);
noiseanom=kse_test(noiseanom);

Index=[1:1:sizeofD];
Index=[Index',zeros(sizeofD,1)];

for i=1:sizeofD
    if Output(i)>0.7
       Index(i,2)=1;
    end
end
ACTUAL=Index;
Origindex=find(ACTUAL(:,2)==1);

Index=[1:1:sizeofD];
Index=[Index',zeros(sizeofD,1)];

for i=1:sizeofD
    if noiseunif(i)>0.7
       Index(i,2)=1;
    end
end
Uniformm=Index;
Pertindex=find(Uniformm(:,2)==1);



Index=[1:1:sizeofD];
Index=[Index',zeros(sizeofD,1)];

for i=1:sizeofD
    if noiseanom(i)>0.7
       Index(i,2)=1;
    end
end
Anomm=Index;
 Pertindex1=find(Anomm(:,2)==1);
IND1=Anomm;
IND=Uniformm;
% Evaluate(ACTUAL,IND)
C=union(Origindex,Pertindex);
C=size(C,1);
p = size(Origindex,1);
n = sizeofD-p;
N = p+n;
tp = sum(ACTUAL(Origindex,2)==IND(Origindex,2));
tn = N-C;
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
EVAL(rr,:) = [accuracy tp specificity precision recall f_measure gmean]+EVAL(rr,:);


C=union(Origindex,Pertindex1);
C=size(C,1);
p = size(Origindex,1);
n = sizeofD-p;
N = p+n;
tp = sum(ACTUAL(Origindex,2)==IND1(Origindex,2));
tn = N-C;
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
EVAL1(rr,:) =[accuracy tp specificity precision recall f_measure gmean]+EVAL1(rr,:);
end
EVAL(rr,:)=EVAL(rr,:)/j;
EVAL1(rr,:)=EVAL1(rr,:)/j;
end

