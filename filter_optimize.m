%% 2024-6-1 params_0601
%load('.\save-mat\params_0504.mat');
%load ('.\save-mat\params_0401.mat');
%save('.\save-mat\params_0601.mat');
clear all;
load ('.\save-mat\params_0601.mat');
% x_fir2 = x_fir;
% save ('.\save-mat\params_0601.mat');
%x_fir:每次迭代乘，x_fir2:最后乘

%% 2024-5-31 2D滤波前传-函数功能测试
clear all;
close all;
load ('.\save-mat\params_0601.mat');
%
nls = 0.05;
map_level = nls;
read_level = nls;
% handle_2d = 1;
h_rgb = im2double(imread('.\pre-pic\6-5.png'));
choice = [1,0,0,1];
Nx = 16;
orth_flag = 1;
tic;
[PSNR_rgb,~,hat_h_rgb] = amp_2d_rgb_filt(N,h_rgb,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,Gauss_noise_map,Gauss_noise_read);
toc;
PSNR_rgb

%% 2024-5-31 dct差值观察
% dct_ratio_1 = dct_ratio_cal(hat_h_rgb(:,:,:,4),hat_h_rgb(:,:,:,1));
% imagesc(log10(abs(dct_ratio_1)));colorbar;
figure(1);
Nx1 = 1;
Nx2 = 32;
subplot(1,3,1);
imshow(dct2_rgb(hat_h_rgb(:,:,:,1)));
xlim([Nx1 Nx2]);
subplot(1,3,2);
imshow(dct2_rgb(hat_h_rgb(:,:,:,4)));
xlim([Nx1 Nx2]);
subplot(1,3,3);
imshow(dct2_rgb(hat_h_rgb(:,:,:,4))-dct2_rgb(hat_h_rgb(:,:,:,1)));
xlim([Nx1 Nx2]);

figure(2);
subplot(1,3,2);
imshow(hat_h_rgb(:,:,:,1));
subplot(1,3,3);
imshow(hat_h_rgb(:,:,:,4));
subplot(1,3,1);
imshow(h_rgb)

%% 2024-6-1 2D滤波前传-迭代次数
clear all;
close all;
load ('.\save-mat\params_0601.mat');
%
nls = 0.05;
map_level = nls;
read_level = nls;
% handle_2d = 1;
h_rgb = im2double(imread('.\pre-pic\6-0.png'));

nums = 20;
%nums = [1:1:4,5:5:100];
L = length(nums);
PSNR_rgb = zeros(1,L);
hat_h_rgb = zeros(N,N,3,L);
PSNR_rgb_noise = zeros(1,L);
hat_h_rgb_noise = zeros(N,N,3,L);
PSNR_rgb_filt = zeros(1,L);
hat_h_rgb_filt = zeros(N,N,3,L);
Nx = 16;
for i = 1:1:L
    num = nums(i);
    tic;
    choice = [1,0,0,1];
    x_fir = ones(N,1);
    [PSNR_rgb_now,~,hat_h_rgb_now] = amp_2d_rgb_filt(N,h_rgb,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,Gauss_noise_map,Gauss_noise_read);
    
    PSNR_rgb(i) = PSNR_rgb_now(1);
    hat_h_rgb(:,:,:,i) = hat_h_rgb_now(:,:,:,1);    
    
    PSNR_rgb_noise(i) = PSNR_rgb_now(4);
    hat_h_rgb_noise(:,:,:,i) = hat_h_rgb_now(:,:,:,4);
    
    choice = [0,0,0,1];
    Nt = 96;
    At = 0.00;
    x_fir = [ones(Nt,1);At*ones(N-Nt,1)];
    [PSNR_rgb_filt_now,~,hat_h_rgb_filt_now] = amp_2d_rgb_filt(N,h_rgb,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,Gauss_noise_map,Gauss_noise_read);
    PSNR_rgb_filt(i) = PSNR_rgb_filt_now(4);
    hat_h_rgb_filt(:,:,:,i) = hat_h_rgb_filt_now(:,:,:,4); 
    
    fprintf('%d in %d ',i,L);
    toc;
end

%% 2024-6-1 观察单张
c = L;
close all;
figure;
hold on;
subplot(1,3,1);
imshow(hat_h_rgb(:,:,:,c));
title(sprintf('noiseless PSNR=%.2fdB',PSNR_rgb(c)),'Fontsize',12);
subplot(1,3,2);
imshow(hat_h_rgb_noise(:,:,:,c));
title(sprintf(['nl=',num2str(nls),' PSNR=%.2fdB'],PSNR_rgb_noise(c)),'Fontsize',12);
subplot(1,3,3);
imshow(hat_h_rgb_filt(:,:,:,c));
title(sprintf(['nl=',num2str(nls),' with filt, PSNR=%.2fdB'],PSNR_rgb_filt(c)),'Fontsize',12);


%% 2024-6-1 迭代次数观察
close all;
figure(1);
plot(nums,PSNR_rgb,'-*','LineWidth',1.2);
hold on;
plot(nums,PSNR_rgb_noise,'-s','LineWidth',1.2);
plot(nums,PSNR_rgb_filt,'-o','LineWidth',1.2);
xlim([-5,105]);xticks(0:20:100);
ylim([8,32]);
grid;
%legend('Noiseless','2% Noise','FontSize',10);
legend('Noiseless','5% Noise','5% Noise with filt','FontSize',10);
xlabel('iteration number','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
title('PSNR vs iteration number','FontSize',12);

%% 2024-6-1 2D滤波-扫描Nx寻找最优Nx（5张图分别作）
clear all;
close all;

load ('.\save-mat\params_0601.mat');
%
nls = 0.05;
map_level = nls;
read_level = nls;
% handle_2d = 1;
c = 4;
h_rgb = im2double(imread(['.\pre-pic\6-',num2str(c),'.png']));

PSNRs_noise = [22.30,19.36,20.33,20.74,20.25];
PSNRs_noiseless = [32.77,29.81,30.92,26.47,24.07];
%Nxs = [1,8:8:256];
Nxs = 1:1:8;
L = length(Nxs);
PSNRs_filt = zeros(1,L);
hat_hs_filt = zeros(N,N,3,L);

for i = 1:1:L
    Nx = Nxs(i);
    tic;
    
    choice = [0,0,0,1];
    Nt = 128;
    At = 0.01;
    x_fir = [ones(Nt,1);At*ones(N-Nt,1)];
    [PSNR_rgb_filt_now,~,hat_h_rgb_filt_now] = amp_2d_rgb_filt(N,h_rgb,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,Gauss_noise_map,Gauss_noise_read);
    PSNRs_filt(i) = PSNR_rgb_filt_now(4);
    hat_hs_filt(:,:,:,i) = hat_h_rgb_filt_now(:,:,:,4); 
    
    fprintf('%d in %d ',i,L);
    toc;
end

save(['.\save-mat\psnrs_0601_Nxs_',num2str(c),'_finetune.mat'],'PSNRs_filt','hat_hs_filt');

%% 2024-6-1 上个实验的画图
close all;
plot(Nxs,PSNRs_filt,'*-','LineWidth',1.5);
grid;
hold on;
plot(Nxs,PSNRs_noise(c+1)*ones(1,length(Nxs)),'--','LineWidth',1.5);
plot(Nxs,PSNRs_noiseless(c+1)*ones(1,length(Nxs)),'--','LineWidth',1.5);
xlabel('Nx','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
legend('5% Noise with filter','5% Noise','Noiseless','FontSize',10);
%xlim([Nxs(1)-8 Nxs(end)+8]);xticks(0:16:256);
xlim([Nxs(1)-1 Nxs(end)+1]);xticks(0:1:8);
ylim([PSNRs_noise(c+1)-2  PSNRs_noiseless(c+1)+2])
title('2D AMP reconstruction PSNR-N_x curve','FontSize',12);

%% 2024-6-1 2D滤波-扫描Nt寻找最优Nt（5张图分别作）
clear all;
close all;
load ('.\save-mat\params_0601.mat');
nls = 0.05;
map_level = nls;
read_level = nls;
% handle_2d = 1;
c = 4;
h_rgb = im2double(imread(['.\pre-pic\6-',num2str(c),'.png']));

PSNRs_noise = [22.30,19.36,20.33,20.74,20.25];
PSNRs_noiseless = [32.77,29.81,30.92,26.47,24.07];
Nx = 8;
Nts = 8:8:256;

L = length(Nts);
PSNRs_filt = zeros(1,L);
hat_hs_filt = zeros(N,N,3,L);

for i = 1:1:L
    tic;
    choice = [0,0,0,1];
    Nt = Nts(i);
    At = 0.01;
    x_fir = [ones(Nt,1);At*ones(N-Nt,1)];
    [PSNR_rgb_filt_now,~,hat_h_rgb_filt_now] = amp_2d_rgb_filt(N,h_rgb,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,Gauss_noise_map,Gauss_noise_read);
    PSNRs_filt(i) = PSNR_rgb_filt_now(4);
    hat_hs_filt(:,:,:,i) = hat_h_rgb_filt_now(:,:,:,4); 
    
    fprintf('%d in %d ',i,L);
    toc;
end

save(['.\save-mat\psnrs_0601_Nts_',num2str(c),'.mat'],'PSNRs_filt','hat_hs_filt');

%% 2024-6-1 上个实验的画图
close all;
plot(Nts,PSNRs_filt,'*-','LineWidth',1.5);
grid;
hold on;
plot(Nts,PSNRs_noise(c+1)*ones(1,length(Nts)),'--','LineWidth',1.5);
plot(Nts,PSNRs_noiseless(c+1)*ones(1,length(Nts)),'--','LineWidth',1.5);
xlabel('Nt','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
legend('5% Noise with filter','5% Noise','Noiseless','FontSize',10);
xlim([Nts(1)-8 Nts(end)+8]);xticks(Nts(1):16:Nts(end));
%xlim([Nts(1)-1 Nts(end)+1]);xticks(Nts(1):16:Nts(end));
ylim([PSNRs_noise(c+1)-2  PSNRs_noiseless(c+1)+2])
title('2D AMP reconstruction PSNR-N_t curve','FontSize',12);


%% 2024-6-1 在最优的Nt和Nx下分别各自重建图像，并保存重建后的图像（无噪重建，有噪重建，有噪+滤波）
clear all;
close all;
load ('.\save-mat\params_0601.mat');
nls = 0.05;
map_level = nls;
read_level = nls;
% handle_2d = 1;
% PSNRs_noise = [22.30,19.36,20.33,20.74,20.25];
% PSNRs_noiseless = [32.77,29.81,30.92,26.47,24.07];
source_root = '.\pre-pic\filter\';
images = dir(fullfile(source_root,'*.png'));
L = length(images);

PSNRs_noise = zeros(1,L);
PSNRs_noiseless = zeros(1,L);
PSNRs_filt = zeros(1,L);

Nts = [56,48,40,56,88];
%Nts = [56,56,56,56,56];
Nxs = [8,8,8,8,8];
At = 0.01;

for i = 1:1:L
    
    pic_name = images(i).name;   
    h = im2double(imread([source_root,pic_name]));
    
    Nt = Nts(i);
    Nx = Nxs(i);
    
    tic;
    choice = [1,0,0,1];
    x_fir = ones(N,1);
    [PSNR_rgb_now,~,hat_h_rgb_now] = amp_2d_rgb_filt(N,h,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,Gauss_noise_map,Gauss_noise_read);
    
    PSNRs_noiseless(i) = PSNR_rgb_now(1);
    PSNRs_noise(i) = PSNR_rgb_now(4);
    imwrite(hat_h_rgb_now(:,:,:,1),[source_root,'noiseless\',pic_name]);
    imwrite(hat_h_rgb_now(:,:,:,4),[source_root,'noise\',pic_name]);
    
    choice = [0,0,0,1];
    x_fir = [ones(Nt,1);At*ones(N-Nt,1)];
    [PSNR_rgb_filt_now,~,hat_h_rgb_filt_now] = amp_2d_rgb_filt(N,h,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,Gauss_noise_map,Gauss_noise_read);
    PSNRs_filt(i) = PSNR_rgb_filt_now(4);
    imwrite(hat_h_rgb_filt_now(:,:,:,4),[source_root,'noise_filted\',pic_name]);
    
    fprintf('%d in %d,noiseless = %.2fdB,noise = %.2fdB,noise_filt = %.2fdB,',i,L,PSNRs_noiseless(i),PSNRs_noise(i),PSNRs_filt(i));
    toc;
    
end


save('.\save-mat\psnrs_0601_optim_Nt_Nx.mat','PSNRs_noise','PSNRs_noiseless','PSNRs_filt','Nt','Nx');



%% 2024-6-1 画图
clear all;
close all;
clc;
load('.\save-mat\psnrs_0601_optim_Nt_Nx.mat');

close all;
plot(PSNRs_noiseless,'*-','LineWidth',1.5);
grid;
hold on;
plot(PSNRs_noise,'s-','LineWidth',1.5);
plot(PSNRs_filt,'o-','LineWidth',1.5);
xlabel('image index','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
legend('noiseless','5% noise','5% noise with filter','FontSize',10);
xlim([0.5 5.5]);xticks(1:1:5);
ylim([18 34])
title('2D-AMP reconstruction','FontSize',12);

%% ====================== Function Below ====================== %%
%% amp_eta_t 2024-2-2
function eta = amp_eta_t(r,lambda)
    %[a,b] = size(r);
    %scalar function eta(r,l) = sgn(r) * max(|r|-l,0)
    sgn1 = sign(r);
    minus1 = abs(r) - lambda;
    minus1(minus1<0) = 0;
    eta = sgn1 .* minus1;
end

%% pic2double 2024-2-2
% transform any picture h to N*N double matrix(gray picture)
function [h_double] = pic2double(h,N)
    [N1,N2,color] = size(h);
    if(color==3)
        h1 = rgb2gray(h);
    else
        h1 = h;
    end
    h2 = im2double(h1);
    
    if((N1~=N)&&(N2~=N))
        h_double = imresize(h2,[N,N]);
    else
        h_double = h2;
    end
end

%% fir_self 2024-4-1
function [b,filtered] = fir_self(a,fir_i)
    %a:original signal
    %N:stage
    %fc:cut off frequency in (0,1),1 for nyquist
    %fir1(n,fc,'ftype',window),ftype=high to design high-pass,
    %window default=hamming
    N = fir_i(1);
    fc = fir_i(2);
    type = fir_i(3);
    flag = fir_i(4);
    
    if(flag == 0)
        filtered = a;
        b = 1;
    else
        if(type == 0)
            b = fir1(N,fc,rectwin(N+1));
        elseif(type == 1)
            b = fir1(N,fc,hamming(N+1));
        elseif(type == 2)
            b = fir1(N,fc,hann(N+1));
        elseif(type == 3)
            b = fir1(N,fc,blackman(N+1));
        elseif(type == 4)
            b = fir1(N,fc,bartlett(N+1));
        elseif(type == 5)
            b = fir1(N,fc,kaiser(N+1));
        elseif(type == 6)
            b = fir1(N,fc,chebwin(N+1));
        else
            error('This type has not been developed.');
        end
        
        % filter handle
        filtered = filter(b,1,a);
        
        % phase change
        [n,~] = size(a);
        if(n>1)
            filtered = [filtered(N/2+1:end);a(end-N/2+1:end)];
        else
            filtered = [filtered(N/2+1:end),a(end-N/2+1:end)];
        end
    end
end

%% fetch all subfolders 2024-3-20
function foldersCell = getAllSubfolders(folderPath)
    % 获取指定路径下的所有内容
    items = dir(folderPath);
    
    % 过滤出所有的子文件夹（排除'.'和'..'）
    folders = items([items.isdir] & ~ismember({items.name}, {'.', '..'}));
    
    % 提取所有子文件夹的名称
    foldersNames = {folders.name};
    
    % 将文件夹名称保存到cell数组中
    foldersCell = foldersNames;
end

%% 展一个生成随机数的函数，输入M,N，生成两种矩阵库 2024-5-4 
function [Gauss_noise_map,Gauss_noise_read] = iters_generate(M,N,seed)
    Gauss_noise_read = zeros(M,N,100);
    rng(seed);
    Gauss_noise_map = randn(M,N);
    rng('shuffle');
    for i = 1:1:100
        Gauss_noise_read(:,:,i) = randn(M,N);
    end
end

%% meas-matrix and trans_matrix generate 2024-4-21
function Phi = meas_matrix_generate(M,N,d,flag,orth_flag)
    % Gauss
    if flag == 1
        rng(11637);
        Phi = randn(M,N) * sqrt(1/M);
    % 0,1
    elseif flag == 2
        rng(11637);
        Phi = zeros(M,N);
        for i = 1:1:N
            col_idx = randperm(M);
            Phi(col_idx(1:d),i) = 1/sqrt(d);
        end
    % Bernouli
    elseif flag == 3
        rng(11637);
        Phi = randi([0,1],M,N);
        %If your MATLAB version is too low,please use randint instead
        Phi(Phi==0) = -1;
        Phi = Phi/sqrt(M);
    elseif flag == 4
        rng(11637);
        Phi = randi([-1,4],M,N);%If your MATLAB version is too low,please use randint instead
        Phi(Phi==2) = 0;%P=1/6
        Phi(Phi==3) = 0;%P=1/6
        Phi(Phi==4) = 0;%P=1/6
        Phi = Phi*sqrt(3/M);
    elseif flag == 5
        % d在此处=L
        % 
        % 生成p=1/(d+2)的伯努利矩阵
        rng(11637);
        Phi = randi([-1,d],M,N);
        if d < 1
            Phi(Phi==0) = 1;
        elseif d == 1
            %Phi = Phi;
        else
            for i = 2:1:d
                Phi(Phi==i) = 0;
            end
        end
        Phi = Phi * sqrt((d+2) / 2 / M);
    else
        error('Please type in 1,2,or 3');
    end
    
    % orth
    if orth_flag == 1
        Phi = orth(Phi')';
    end
end

function Psi = trans_matrix_generate(N,flag)
    %dct
    if flag == 0
        Psi = eye(N);
    elseif flag == 1
        Psi = dctmtx(N);
    else
        error('Please type in 0 or 1');
    end
end

%% amp_core with trans-domain measurement 2024-3-30

% 1D-AMP for sparse vector (new_version)
function [x,z] = amp_core_filt(y,A,theta,num,epsilon,alpha,fir1,fir2,x_fir,x_fir2,noise_info,stop_earlier,show_detail,noise_map_MN,noise_read_MN)
    %y: M*1 signal after measurement
    %A: Phi * Psi
    %num: max iteration num
    %epsilon: min iteration step
    %alpha: usually alpha=1 for this fucntion
    %noise_info: [map_noise_flag,map_level,read_noise_flag,read_level]...
    %...flag==0 for noiseless
    %theta: ground truth for sparse vector to be reconstructed]
    %stop_earlier: no need for run all num iteration...
    %...if less than epsilon, break.
    
    %x: N*1 sparse signal reconstruction result
    %z: M*1 residual(during iteration)
    %MSE/NMSE: MSE/NMSE change in the iteration process
    
    map_noise_flag = noise_info(1);
    map_noise_level = noise_info(2); % typical, or 1% (optimized)
    read_noise_flag = noise_info(3);
    read_noise_level = noise_info(4); % typical, or 1% (optimized)
    
    MSE = [];
    NMSE = [];
    
    [M,N] = size(A);
    x0 = zeros(N,1);
    z0 = y;
    
    % map deviation
    if map_noise_flag
        %noise = max(abs(A(:)))*map_noise_level*randn(M,N);
        noise = max(abs(A(:)))*map_noise_level*noise_map_MN;
        A_map_noise = A + noise;
    else
        A_map_noise = A;
    end
        % rng('shuffle');
    
    %iteration
    % x0,z0→x,z
    for t = 1:1:num
        
        lambda0 = alpha*norm(z0,2) / sqrt(M);
        
        b0 = sum(x0(:)~=0) / M;   %AMP append element        
        % read deviation
%         if read_noise_flag
%             noise = max(abs(A(:)))*read_noise_level*randn(M,N);
%             A_noise = A_map_noise + noise;
%         else
%             A_noise = A_map_noise;
%         end
        if read_noise_flag
            % noise = max(abs(A(:)))*read_noise_level*randn(M,N);
            noise = max(abs(A(:)))*read_noise_level*noise_read_MN(:,:,t);
            A_noise = A_map_noise + noise;
        else
            A_noise = A_map_noise;
        end

        [~,r0] = fir_self(A_noise' * z0, fir1);
        
        x = amp_eta_t(r0 + x0, lambda0);
        
        x = x.*x_fir;  
        
%         read deviation     
%         if read_noise_flag
%             noise = max(abs(A(:)))*read_noise_level*randn(M,N);
%             A_noise = A_map_noise + noise;
%         else
%             A_noise = A_map_noise;
%         end
        
        [~,s] = fir_self(A_noise * x, fir2);
        
        z = y - s + z0 * b0;
        
        %recording MSE and NMSE change
        MSE_now = (norm(x - theta)^2)/N;
        NMSE_now = (norm(x - theta)^2)/(norm(theta)^2);
        MSE = [MSE,MSE_now];
        NMSE = [NMSE,NMSE_now];
        epsilon_now = norm((x - x0),2) / (norm(x,2)+1e-8);
        
        % show detail
        if show_detail == 1
            fprintf('进行第%d次迭代,epsilon = %4f \n',t,epsilon_now);
        end
        
        % stop earlier
        if(epsilon_now < epsilon)
            if(stop_earlier==1)       
                if show_detail == 1
                    fprintf('在%d次迭代后收敛,结束循环\n',t); 
                end
                break;
            end
        end      
        x0 = x;
        z0 = z;
    end
    
    %变换域滤波
    x = x.*x_fir2;    
end

% 2D-AMP for single real 2D map: N*N theta,compress and recovery
function [hat_theta] = amp_2d_filt(theta,Phi,Psi,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,show_detail,Nx,noise_map_MN,noise_read_MN)
    
    %noise
    noise_info = [0,map_level,0,read_level;
                  1,map_level,0,read_level;
                  0,map_level,1,read_level;
                  1,map_level,1,read_level];
              
    %choice
    sum_choice = sum(choice);
    if((sum_choice<1)||(sum_choice>4))
        error('choice input must in range [0,0,0,1] ~ [1,1,1,1]');
    end
    
    [~,N] = size(Phi);
    hat_theta = zeros(N,N,4);
    
    %Nx = 16;
    
    for noise_tag = 1:1:4
        if(choice(noise_tag)==0)
            continue;
        else
            if handle_2d == 0
                % column-wise handle
                for i = 1:1:N
                    theta_now = theta(:,i);
                    %theta_now = theta(i,:)';
                    x = Psi * theta_now;
                    y = Phi * x;     
                    A = Phi * Psi;       
                    if i == Nx
                        %i
                        x_fir = ones(N,1);
                    end
                    [hat_theta_elem,~] = amp_core_filt(y,A,theta_now,num,epsilon,alpha,fir1,fir2,x_fir,x_fir2,noise_info(noise_tag,:), ...
                                                  1,show_detail,noise_map_MN,noise_read_MN);               
                    hat_theta(:,i,noise_tag) = hat_theta_elem;
                end
            elseif handle_2d == 1
                % row-wise handle
                for i = 1:1:N
                    theta_now = theta(i,:)';
                    x = Psi * theta_now;
                    y = Phi * x;     
                    A = Phi * Psi;        
                    if i == Nx
                        x_fir = ones(N,1);
                    end
                    [hat_theta_elem,~] = amp_core_filt(y,A,theta_now,num,epsilon,alpha,fir1,fir2,x_fir,x_fir2,noise_info(noise_tag,:), ...
                                                  1,show_detail,noise_map_MN,noise_read_MN);               
                    hat_theta(i,:,noise_tag) = hat_theta_elem';
                end
            else
                error('This handle has not been developed.');
            end
        end
    end
end

% 2D-AMP for single channel picture compression and recovery
function [PSNR,hat_theta,hat_h] = amp_2d_gray_filt(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,noise_map_MN,noise_read_MN)
    
    h = pic2double(h,N);
    M = round(N * compress_rate); 
    
    %Gauss
    rng(11637);
    Phi = randn(M,N); 
    Phi = sqrt(1/M) * Phi;
    
    %bernouli
%     rng(11637);
%     orth_flag = 0;
%     Phi = randi([0,1],M,N);
%     %If your MATLAB version is too low,please use randint instead
%     Phi(Phi==0) = -1;
%     Phi = Phi/sqrt(M);

    if orth_flag == 1
        Phi = orth(Phi')';
    end
    
    
    Psi = eye(N,N);
    
    % record result
    hat_h = zeros(N,N,4);
    PSNR = zeros(1,4);
    
    %trans_type
    if trans_type == 'dct'
        % dct sparse
        theta = dct2(h);
        hat_theta = amp_2d_filt(theta,Phi,Psi,num,epsilon,alpha,handle_2d,...
                    map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,show_detail,Nx,noise_map_MN,noise_read_MN);
        % record PSNR
        for noise_tag = 1:1:4
            if(choice(noise_tag)==0)
                continue;
            else
                hat_h(:,:,noise_tag) = idct2(hat_theta(:,:,noise_tag));
            end
            PSNR(noise_tag) = psnr(hat_h(:,:,noise_tag),h);     
        end
    elseif trans_type == 'non'
        % origin sparse
        theta = h;
        hat_theta = amp_2d_filt(theta,Phi,Psi,num,epsilon,alpha,handle_2d,...
                    map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,show_detail,Nx,noise_map_MN,noise_read_MN);
        % record PSNR
        for noise_tag = 1:1:4
            if(choice(noise_tag)==0)
                continue;
            else
                hat_h(:,:,noise_tag) = (hat_theta(:,:,noise_tag));
            end      
            PSNR(noise_tag) = psnr(hat_h(:,:,noise_tag),h);
        end
    else
        error('This trans_type has not been developed.');
    end
    
    % show PSNR detail
    formatspec_tot = cell(4,1);
    formatspec_tot{1} = 'noiseless重构PSNR = %4f \n';
    formatspec_tot{2} = 'only map noise重构PSNR = %4f \n';
    formatspec_tot{3} = 'only read noise重构PSNR = %4f \n';
    formatspec_tot{4} = 'map+read noise重构PSNR = %4f \n';
    for j=1:1:4
        if(choice(j)==1 && show_detail==1)
            fprintf(formatspec_tot{j},PSNR(j));
        end 
    end   
end

% 2D-AMP for 3-channel RGB picture compression and recovery
function [PSNR,hat_theta,hat_h] = amp_2d_rgb_filt(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,noise_map_MN,noise_read_MN)
    % need to be rgb
    [~,~,color] = size(h);
    if(color~=3)
        error('Picture Must be RGB three channel.')
    end
    % record result
    hat_theta = zeros(N,N,3,4);
    hat_h = zeros(N,N,3,4);
    PSNR_now = zeros(1,3,4);
    % 3 channels reconstruction
    for i = 1:1:3
        % sum(x_fir(:))
        h_now = h(:,:,i);
        [PSNR_now(:,i,:),hat_theta(:,:,i,:),hat_h(:,:,i,:)] = amp_2d_gray_filt(N,h_now,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,noise_map_MN,noise_read_MN);
    end
    % PSNR for rgb: single average
    PSNR = mean(PSNR_now,2);
end

%% total_dataset_reconstruction and evaluate[gray or rgb] 2024-3-30
% dataset reconstruction
function [PSNRs] = amp_2d_recons_filt(N,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
                   read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nxs,noise_map_MN,noise_read_MN,source_root,target_root,class_name,begin_class,end_class,save_flag)
     
    PSNRs = cell(1,end_class - begin_class + 1);    
    for i = begin_class:1:end_class
        tic;
        class = [(class_name{i}),'\'];
        %create target folder
        if ~(isfolder([target_root,class]))
            disp('target folder doesnt exist,create it.');
            mkdir([target_root,class]);
            disp('successfully create.')
        end
        
        images = dir(fullfile(source_root,class,'*.png'));
        images_num = length(images);
        
        c_indexes = find(choice == 1);
        c_index = c_indexes(1);
        disp(c_index);
        
        i_real = i - begin_class + 1;
        PSNRs{i_real} = [];
        for j = 1:1:images_num
            % read
            pic_name = images(j).name;   
            h = imread([source_root,class,pic_name]);
            [~,~,color] = size(h);
            % recons and save
            Nx = Nxs(j);
            if color == 1
                [PSNR,~,hat_h] = amp_2d_gray_filt(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,noise_map_MN,noise_read_MN);
                if save_flag == 1                   
                    imwrite(hat_h(:,:,c_index),[target_root,class,pic_name]);
                end                
            elseif color == 3
                [PSNR,~,hat_h] = amp_2d_rgb_filt(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,x_fir2,trans_type,show_detail,Nx,noise_map_MN,noise_read_MN);
                if save_flag == 1                   
                    imwrite(hat_h(:,:,:,c_index),[target_root,class,pic_name]);
                end
            else
                error('Channels must be 1 or 3');
            end
            % print
            fprintf('class %s recons-progress : %d in %d ,PSNR = %2f,class: %d in %d\n',class_name{i},j,images_num,PSNR(c_index),i-begin_class,end_class-begin_class);
            PSNRs{i_real} = [PSNRs{i_real},PSNR(c_index)];
        end
        toc;
    end
end

% cell2vector and evaluate PSNR
function [mus,sigmas,PSNR_vec] = PSNR_evaluate(PSNRs)
    PSNR_vec = [];
    class_num = length(PSNRs);
    % cell to vector
    for i=1:1:class_num
        PSNR_vec = [PSNR_vec,PSNRs{i}];
    end
    mus = mean(PSNR_vec);
    sigmas = std(PSNR_vec);
end

% evaluate statistical
function [] = PSNR_stat(PSNR_vec,PSNR_vec_noise)
   
    N_bins = 50;
    fontsize = 15;
    labelsize = 13;

    figure;
    histogram(PSNR_vec,N_bins,'FaceColor',"#EDB120");
    xlim([10,50]);
    xlabel('PSNR/dB','FontSize',labelsize);ylabel('frequency','FontSize',labelsize);

    hold on;
    histogram(PSNR_vec_noise,N_bins,'FaceColor',"#0072BD");
    title('PSNR distribution','FontSize',fontsize);
    % legend('noiseless','map+read noise','FontSize',10);
    legend('sim','test','FontSize',10);
    grid;
end

%% 计算MRM值 2024-5-12
function [ksi] = mrm_cal(A1,An)
    % 计算奇异值
    sv_A1 = svd(A1);
    sv_An = svd(An);
    
    ksi = max(sv_A1) / min(sv_An);

end

%% 计算rgb dct
function [h_dct] = dct2_rgb(h)
    [M,N,color] = size(h);
    if color == 3
        h_dct = zeros(M,N,3);
        for i = 1:1:3
           h_dct(:,:,i) = dct2(h(:,:,i)); 
        end
    else
        error('channel of h must be 3');
    end
end

%% 计算dct的比值 2024-5-30
function [dct_ratio] = dct_ratio_cal(P1,P2)
    P1_1 = im2double(P1);
    P2_1 = im2double(P2);
    [~,~,color] = size(P1_1);
    if color == 1
        dct_ratio = dct2(P1_1)./dct2(P2_1);
        
    elseif color == 3
        dct_ratio_3 = dct2_rgb(P1_1)./dct2_rgb(P2_1);
%         for i = 1:1:3
%             dct_ratio_3(:,:,i) = dct2(P1_1(:,:,i))./ (dct2(P2_1(:,:,i) + 1e-4*ones(M,N)));
%         end
        dct_ratio = mean(dct_ratio_3,3);
    end
end
