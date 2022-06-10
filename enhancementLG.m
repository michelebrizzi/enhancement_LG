function F = enhancementLG(img, C)
% Input:
% -img: image to enhance
% -C: enhancement factor (usually in the range 1.1-1.5)
scala = [14.405, 7.2025, 3.6013, 1.8006, 0.9003];

[nrsize, ncsize, ~]=size(img);
if mod(nrsize,2)
    img=img(1:end-1,:,:);
end
if mod(ncsize,2)
    img=img(:,1:end-1,:);
end

[nrsize, ncsize, ~]=size(img);
map = C * ones(nrsize, ncsize);
[mat_LG, denom, HH] = prepare_LG_functions(img, scala);
F = enhancement_LG_multiscale(img, 'log', map, mat_LG, denom, scala, HH);

figure, imshowpair(img ,F, 'montage')
end

% create LG filters for that image size (for repeated use)
function [mat_LG, denom, HH] = prepare_LG_functions(img, scala)

[nrsize, ncsize, ~]=size(img);

n=1; k=0;
denom=zeros(nrsize, ncsize);
mat_LG=cell(1,length(scala));
for ns=1:length(scala)
    f_LG = LG_filter(nrsize, ncsize, n, k, scala(ns), 1);
    fft_LG = fft2(f_LG, nrsize, ncsize);
    mat_LG{ns} = fft_LG;
    denom=denom+fft_LG.*conj(fft_LG);
end
denom=denom+0.001*max(max(denom));

scala0=40;
padding=[10*ceil(nrsize/(10))+mod(ceil(nrsize/(10)),2),10*ceil(ncsize/(10))+mod(ceil(ncsize/(10)),2)];
hh=fspecial('gaussian', [nrsize+2*padding(1) ncsize+2*padding(2)], scala0);
HH=fft2(hh);
end

% Enhancement
function F = enhancement_LG_multiscale(img, domain, map, mat_LG, denom, scala, HH)

img=im2double(img);
YIQ=rgb2ntsc(img);

Y=YIQ(:,:,1);
if strcmp(domain, 'log')
    Y=log(Y+1);
end

[nrsize, ncsize]=size(Y);
padding=[10*ceil(nrsize/(10))+mod(ceil(nrsize/(10)),2),10*ceil(ncsize/(10))+mod(ceil(ncsize/(10)),2)];

Ypad=padarray(Y,padding,'replicate','both');
fft_Y=fft2(Ypad);
LP=fftshift(ifft2(fft_Y.*HH));
LPY=LP(1+padding(1):size(Y,1)+padding(1),1+padding(2):size(Y,2)+padding(2));
Y=Y-LPY;
fft_Y=fft2(Y);

I=YIQ(:,:,2);
Ipad=padarray(I,padding,'replicate','both');
fft_I=fft2(Ipad);
LP=fftshift(ifft2(fft_I.*HH));
LPI=LP(1+padding(1):size(I,1)+padding(1),1+padding(2):size(I,2)+padding(2));
I=I-LPI;
fft_I=fft2(I);

Q=YIQ(:,:,3);
Qpad=padarray(Q,padding,'replicate','both');
fft_Q=fft2(Qpad);
LP=fftshift(ifft2(fft_Q.*HH));
LPQ=LP(1+padding(1):size(Q,1)+padding(1),1+padding(2):size(Q,2)+padding(2));
Q=Q-LPQ;
fft_Q=fft2(Q);

%Y
out=zeros(nrsize, ncsize);
for ns = 1:length(scala)
    fft_edge = fft_Y.*mat_LG{ns};
    edges = ifft2(fft_edge);
    fft_edge = fft2(1/scala(ns).*fftshift(map).*edges);
    out_f = ifft2(fft_edge .* conj(mat_LG{ns})./denom);
    out = out + scala(ns).*out_f;
end
out = real(out);
YY = out+LPY;

%I
out = zeros(nrsize, ncsize);
for ns = 1:length(scala)
    fft_edge = fft_I.*mat_LG{ns};
    edges = ifft2(fft_edge);
    fft_edge = fft2(1/scala(ns).*fftshift(map/1.01).*edges);
    out_f = ifft2(fft_edge .* conj(mat_LG{ns})./denom);
    out = out + scala(ns).*out_f;
end
out = real(out);
II=out+LPI;

%Q
out = zeros(nrsize, ncsize);
for ns = 1:length(scala)
    fft_edge = fft_Q.*mat_LG{ns};
    edges = ifft2(fft_edge);
    fft_edge = fft2(1/scala(ns).*fftshift(map/1.01).*edges);
    out_f = ifft2(fft_edge .* conj(mat_LG{ns})./denom);
    out = out + scala(ns).*out_f;
end
out = real(out);
QQ=out+LPQ;

if strcmp(domain, 'log')
    YY=exp(YY)-1;
end

% figure, imshowpair(Y,YY,'montage')
% figure, imshowpair(I,II,'montage')
% figure, imshowpair(Q,QQ,'montage')

YYIIQQ=cat(3,YY,II,QQ);
F=ntsc2rgb(YYIIQQ);

%figure, imshowpair(img,F,'montage')
end

%Laguerre-Gauss filters
function Lrt = LG_filter(num_r, num_c, n, k, scala, type)
% Funzioni di Laguerre-Gauss di ordine n,k

nr=floor(num_r/2);
nc=floor(num_c/2);
t_uc= ones(1,num_c);
t_ur= ones(1,num_r);
ir=nr:-1:-(nr-1);
ic=-nc:(nc-1);

rho     = sqrt((ir.^2).'*t_uc+((ic.^2).'*t_ur).')/scala;
theta	= atan2(-(ir.'*t_uc),-(ic.'*t_ur).');

if type
    % Rodriguez formula
    L=zeros(num_r, num_c, k+1);
    for h=0:k
        L(:,:,h+1)=(((-1).^h).*(factorial(n+k)./(factorial(k-h).*factorial(n+h))).*(((2*pi*rho.^2).^h)./factorial(h)));
    end
    LL=sum(L,3);
    Lrt=((-1).^k).*(2^(n+1)./2).*(pi.^(n./2)).*sqrt(factorial(k)./factorial(n+k))./scala.*(rho.^n).*exp(-pi*(rho.^2)).*LL.*exp(1i.*n.*theta);
else
    % Rodriguez formula
    L=zeros(num_r, num_c, k+1);
    t=(rho.^2)./(8*pi^3);
    for h=0:k
        L(:,:,h+1)=(((-1).^h).*(factorial(n+k)./(factorial(k-h).*factorial(n+h))).*(((t).^h)./factorial(h)));
    end
    LL=sum(L,3);
    cnk=((-1).^k.*(-1i).^n).*sqrt(factorial(k)./factorial(n+k))./((2^(n-1)./2).*(pi.^(n./2)));
    Lrt=cnk./scala.*((rho./(2*pi)).^n).*exp(-(rho.^2)./(16*pi^3)).*LL.*exp(1i.*n.*theta);
    %     Lrt=fftshift(Lrt);
end
% Lrt=Lrt./max(max(abs(Lrt)));
end