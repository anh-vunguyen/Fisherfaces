%% Auteur : NGUYEN Anh Vu
%% Instructeur :  Oscar ACOSTA
%% FACE RECOGNITION WITH THE METHOD OF FISHERFACES (PCA + LDA)
%% Reference: P. Belhumeur, J. Hespanha, and D. Kriegman, “Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection,”
%% IEEE Trans. Pattern Anal. Mach.Intell., vol. 19, no. 7, pp. 711-720, 1997.

close all
clc

%% Load Image
load('ImDatabase.mat');

%% Useful coefficients
% Number of class
NoC = 30;
% Number of images per class
NoI = 15;
% Dimension of each image
dimImg = 32256; 
% 50 largest eigen vectors
nb = 50;


%% Mean image of all samples
MeanImg = sum(A,2)/size(A,2);

% % Show mean image of all samples
% ShowMeanImg = reshape(MeanImg,192,168);
% figure (1)
% imshow(ShowMeanImg);
% title('Mean Image of all samples');

%% Mean image of each class
MImgEC = zeros(dimImg,NoC);
for i = 1:1:NoC
    sumTmp = sum(A(:,(i-1)*15+1:i*15),2);
    MImgEC(:,i) = sumTmp/NoI;
end

% Test show mean image of class 1
% figure (2)
% ShowMeanImg = reshape(MImgEC(:,1),192,168);
% imshow(ShowMeanImg);
% title('Mean Image of class 1');

% BECAUSE OF THE HIGH DIMENSION MATRIX AND THE LIMIT OF STORAGE, WE MUST
% APPLY PCA METHOD TO REDUCE THE DIMENSION OF MATRIX

%% Covariance Matrix
x = bsxfun(@minus,A,MeanImg);
s = x'*x;

%% Eigen vectors and Eigen values
[vectorC, valueC] = eig(s);
ss = diag(valueC);
[ss,iii] = sort(-ss);
vectorC = vectorC(:,iii);
vectorL = x*vectorC(:,1:nb); % Take 50 largest eigen vectors
Coeff= x'*vectorL;

%% Mean coefficients of all samples
MeanCoeffA = sum(Coeff)/size(Coeff,1);

%% Transposed Matrix (to follow research papers)
MeanCoeffA = MeanCoeffA';
Coeff = Coeff';

%% Mean coefficients of each class
MCoeffEC = zeros(nb,NoC); % 50 largest eigen vectors
for i = 1:1:NoC
    MCoeffEC(:,i) = mean(Coeff(:,NoI*(i-1)+1:NoI*i),2);
end

%% Between-class scatter matrix Sb
Sb  = zeros(nb,nb);
for i=1:1:NoC
    Sb = Sb + NoI*(MCoeffEC(:,i)-MeanCoeffA)*((MCoeffEC(:,i)-MeanCoeffA))';
end

%% Within-class scatter matrix Sw
Sw  = zeros(nb,nb);
for i=1:1:NoC
    for j=1:1:NoI
        Sw = Sw + (Coeff(:,(i-1)*NoI + j)-MCoeffEC(:,i))*((Coeff(:,(i-1)*NoI + j)-MCoeffEC(:,i)))';
    end
end

%% Find Optimal Projection Matrix 
[Wopt, EVopt, Dopt] = eig(Sb,Sw);

%% Projected Coefficients:
ProjCoeff = Wopt'*Coeff;

% Calculate projected mean of each class:
PMeC = zeros(nb,NoC);

for i = 1:1:NoC
   PMeC(:,i)= mean(ProjCoeff(:,(i-1)*NoI+1:i*NoI),2);
end

%% Test

% Load Input Face
input = 'yaleB01_P00A+035E+40.pgm';
inputImg_8 = imread(input);
inputImg_db = im2double(inputImg_8);

figure ('name', 'Test Image')
imshow(inputImg_db);

inputImg_db  = reshape(inputImg_db,[192*168,1]);
inputImg_db  = bsxfun(@minus,inputImg_db,MeanImg);
inputCoeff = inputImg_db'*vectorL;
inputCoeff = inputCoeff';

ProjInputCoeff = Wopt'*inputCoeff;

% Result
comp_Coeff = zeros(1,NoC);
tmp_res =0;
for i=1:1:NoC
    for j=1:1:nb
       tmp_res = tmp_res + (ProjInputCoeff(j,1)-PMeC(j,i))*(ProjInputCoeff(j,1)-PMeC(j,i)); 
    end
    comp_Coeff(1,i) = tmp_res;
    tmp_res = 0;
end

[M,I] = min(comp_Coeff);
mess1=sprintf('# %d',I);

Org = reshape(A(:,(I-1)*NoI+1),[192,168]);

figure('name','Most Similar Database Image');
imshow(Org);



