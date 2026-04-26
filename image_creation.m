% load image as floating point numbers
img = imread("nyc.jpg");
image = im2double(img);

% ensure divisbility by 8
[m, n] = size(image);
m = m - mod(m, 8);
n = n - mod(n, 8);
image = image(1:m, 1:n);

% apply 8x8 DCT blocks and create a 3D array of values
fun = @(block_struct) dct2(block_struct.data);
B = blockproc(image, [8 8], fun);

% Scale B
min_value = min(B(:));
max_value = max(B(:));
B_scaled = (B - min_value) / (max_value - min_value);

% quantized B
B_quantized = uint8(round(B_scaled * 255));

% Create 3D array
blocks = reshape(B_quantized, [8, m/8, 8, n/8]);
blocks = permute(blocks, [1, 3, 2, 4]);
blocks = reshape(blocks, [8, 8, m*n/64]);

num_blocks = m * n / 64;
B_3d = reshape(B, 8, 8, num_blocks);

% N is the group size being transferred
N = (n*m)/64;

%  DISPLAY THE ACTUAL IMAGE BLOCKS USED IN TRANSMISSION
figure;
sgtitle(sprintf('Original Image Blocks Passed Through Pipeline (N=%d blocks)', N));

for k = 1:N
    % Extract the k-th quantized DCT block and undo quantization -> IDCT
    dct_block_scaled = double(blocks(:,:,k)) / 255 * (max_value - min_value) + min_value;
    pixel_block = idct2(dct_block_scaled);
    pixel_block = max(0, min(1, pixel_block));  % clip to [0,1]

    % subplot(2, N, k);
    % imshow(pixel_block, []);
    % title(sprintf('Block %d (pixels)', k));
    % 
    % % Also show the raw quantized DCT coefficients as an image
    % subplot(2, N, N + k);
    % imshow(blocks(:,:,k), []);
    % title(sprintf('Block %d (DCT)', k));

    fprintf('Block on: %d\n', k);
end

%  ALSO SHOW WHERE THESE BLOCKS COME FROM IN THE FULL IMAGE
figure;
imshow(image, []);
title('Full Image — Highlighted Transmitted Blocks');
hold on;

% The blocks are indexed in column-major order across the image.
% Block index k corresponds to row r, col c in the block grid:
num_block_cols = n / 8;
for k = 1:N
    block_col = ceil(k / (m/8));   % which block-column
    block_row = mod(k-1, m/8) + 1; % which block-row
    % pixel coordinates of top-left corner
    px_x = (block_col - 1) * 8 + 1;
    px_y = (block_row  - 1) * 8 + 1;
    rectangle('Position', [px_x, px_y, 8, 8], ...
              'EdgeColor', 'red', 'LineWidth', 1.5);
end
hold off;

% Bit stream creation
group = blocks(:, :, 1:N);
col_vector = double(group(:));
bit_matrix = int2bit(col_vector, 8);
bit_vector = bit_matrix(:)';

% Modulation
symbols = 2*bit_vector - 1;

% Half-sine pulsing shape function
Ns = 32;
t = linspace(0, 1, Ns+1);
t = t(1:end-1);
g1 = sin(pi*t);
g1 = g1/sqrt(sum(g1.^2)*1/Ns);

% SRRC pulsing shape function
alpha = .5;
K = 6;
g2 = rcosdesign(alpha, 2*K, Ns);
g2 = g2/sqrt(sum(g2.^2)*1/Ns);

% frequency axis (-0.5 to 0.5 normalized frequency)
N_fft = 1024;
f = linspace(-Ns/2, Ns/2, N_fft);

% compute FFTs
g1_f = fftshift(fft(g1, N_fft));
g2_f = fftshift(fft(g2, N_fft));

clf;
numberofbits = length(symbols);
randbitstream = symbols;

randbitstream = upsample(randbitstream, 32);
randbitstream = circshift(randbitstream, 15);
halfsinebits = conv(randbitstream, g1, 'same');
SRRCbits     = conv(randbitstream, g2, 'same');
t_total1 = linspace(0, 10, length(randbitstream));

h = [1,1/2,3/4,-2/7];
heff = [upsample(h, 32), zeros(1, numberofbits*32 - 4*32)];
L = 2^15;
channelhalfsine = conv(heff,halfsinebits);
channelSRRC = conv(heff,SRRCbits);
sigma = .1;
noise = sigma*randn([1,length(channelhalfsine)]);
channelhalfsine = channelhalfsine+noise;
channelSRRC = channelSRRC+noise;
match_g1 = flip(g1);
channelhalfsine = filter(match_g1,1,channelhalfsine);
match_g2 = flip(g2);
channelSRRC = filter(match_g2,1,channelSRRC);
ones = [1, zeros(1, L-1)];
invertedchannel = ifft(1./fft(heff));

H = fft(heff,length(channelhalfsine));
MMSE = conj(H)./(abs(H).^2+sigma^2);
channelhalfsine = ifft(MMSE.*fft(channelhalfsine));
channelSRRC = ifft(MMSE.*fft(channelSRRC));
% receivedhalfsine = channelhalfsine(1:numberofbits*32);
% receivedSRRC     = channelSRRC(1:numberofbits*32);
receivedhalfsine = channelhalfsine;
receivedSRRC     = channelSRRC;

% % For half-sine: find peak of matched filter output on an isolated impulse
% test_impulse = zeros(1, 10*Ns);
% test_impulse(1) = 1;
% hs_response = conv(conv(test_impulse, g1), match_g1);
% [~, peak_hs] = max(abs(hs_response));
% delay_hs = peak_hs - 1;
% 
% figure
% plot(hs_response);
% 
% % For SRRC: same approach
% mmse_imp = real(ifft(MMSE));
% g2_delay = conv(conv(test_impulse, g2), match_g2);
% % srrc_response = conv(conv(test_impulse, g2), g2_delay);
% [~, peak_srrc] = max(abs(g2_delay));
% delay_srrc = peak_srrc - 1;
% % delay_srrc = 384;

sig_len = length(channelhalfsine);
H_full  = fft(heff, sig_len);
MMSE_full = conj(H_full) ./ (abs(H_full).^2 + sigma^2);
mmse_imp  = real(ifft(MMSE_full));  % MMSE impulse response

% Pad test impulse to same length
test_impulse = zeros(1, sig_len);
test_impulse(1) = 1;


% Half-sine: pulse -> matched filter -> MMSE
hs_tx      = conv(test_impulse, g1);
hs_mf      = conv(hs_tx(1:sig_len), match_g1);
hs_eq      = conv(hs_mf(1:sig_len), mmse_imp);
[~, peak_hs] = max(abs(hs_eq(1 : sig_len)));
delay_hs     = peak_hs - 1;

% SRRC: pulse -> matched filter -> MMSE
srrc_tx      = conv(test_impulse, g2);
srrc_mf      = conv(srrc_tx(1:sig_len), match_g2);
srrc_eq      = conv(srrc_mf(1:sig_len), mmse_imp);
[~, peak_srrc] = max(abs(srrc_eq(1 : sig_len)));
% delay_srrc     = peak_srrc - 1;
delay_srrc = 205;

% idx_hs   = delay_hs   + (0:numberofbits-1)*Ns + 1;
% idx_srrc = delay_srrc + (0:numberofbits-1)*Ns + 1;
% 
% idx_hs   = min(idx_hs,   length(receivedhalfsine));
% idx_srrc = min(idx_srrc, length(receivedSRRC));
% 
% sampled_hs   = real(receivedhalfsine(idx_hs));
% sampled_srrc = real(receivedSRRC(idx_srrc));
% 
% detected_hs   = double(sampled_hs   > 0);
% detected_srrc = double(sampled_srrc > 0);

idx_hs   = delay_hs   + (0:numberofbits-1)*Ns + 1;
idx_srrc = delay_srrc + (0:numberofbits-1)*Ns + 1;

% Safety check — warn if any indices exceed signal length
if any(idx_hs > length(receivedhalfsine))
    warning('Half-sine: %d indices exceed signal length', sum(idx_hs > length(receivedhalfsine)));
    idx_hs = idx_hs(idx_hs <= length(receivedhalfsine));
end
if any(idx_srrc > length(receivedSRRC))
    warning('SRRC: %d indices exceed signal length', sum(idx_srrc > length(receivedSRRC)));
    idx_srrc = idx_srrc(idx_srrc <= length(receivedSRRC));
end

% Sample
sampled_hs   = real(receivedhalfsine(idx_hs));
sampled_srrc = real(receivedSRRC(idx_srrc));

% Threshold detection
detected_hs   = double(sampled_hs   > 0);
detected_srrc = double(sampled_srrc > 0);

% Pad to numberofbits if any indices were trimmed
detected_hs(end+1   : numberofbits) = 0;
detected_srrc(end+1 : numberofbits) = 0;

%  CONVERSION TO IMAGE

bit_mat_hs   = reshape(detected_hs,   8, [])';
bit_mat_srrc = reshape(detected_srrc, 8, [])';

px_hs   = uint8(bi2de(bit_mat_hs,   'left-msb'));
px_srrc = uint8(bi2de(bit_mat_srrc, 'left-msb'));

N_blocks_transmitted = N;
blocks_rec_hs   = reshape(px_hs,   [8, 8, N_blocks_transmitted]);
blocks_rec_srrc = reshape(px_srrc, [8, 8, N_blocks_transmitted]);

%  IMAGE POST-PROCESSING

B_rec_hs   = double(blocks_rec_hs)   / 255 * (max_value - min_value) + min_value;
B_rec_srrc = double(blocks_rec_srrc) / 255 * (max_value - min_value) + min_value;

B_rec_2d_hs   = reshape(permute(reshape(B_rec_hs,   [8, 8, 1, N]), [1,3,2,4]), [8, N*8]);
B_rec_2d_srrc = reshape(permute(reshape(B_rec_srrc, [8, 8, 1, N]), [1,3,2,4]), [8, N*8]);

ifun         = @(block_struct) idct2(block_struct.data);
img_out_hs   = blockproc(B_rec_2d_hs,   [8 8], ifun);
img_out_srrc = blockproc(B_rec_2d_srrc, [8 8], ifun);

img_out_hs   = max(0, min(1, img_out_hs));
img_out_srrc = max(0, min(1, img_out_srrc));

% DISPLAY
figure;
subplot(1,2,1);
imshow(img_out_hs, []);
title(sprintf('Half-Sine | MMSE | \\sigma=%.2f', sigma));

subplot(1,2,2);
imshow(img_out_srrc, []);
title(sprintf('SRRC | MMSE | \\sigma=%.2f', sigma));

sgtitle(sprintf('Q14: Recovered Image Patch (N=%d blocks)', N));

% BER
BER_hs   = sum(detected_hs   ~= double(bit_vector)) / numberofbits;
BER_srrc = sum(detected_srrc ~= double(bit_vector)) / numberofbits;
fprintf('SNR           = %.2f dB\n', 10*log10(1/sigma^2));
fprintf('BER Half-Sine : %.6f\n', BER_hs);
fprintf('BER SRRC      : %.6f\n', BER_srrc);

figure;
subplot(1, 3, 1);
imshow(image, []);
title('Original');

subplot(1, 3, 2);
imshow(img_out_hs_full, []);
title(sprintf('Half-Sine | BER=%.4f', BER_hs));

subplot(1, 3, 3);
imshow(img_out_srrc_full, []);
title(sprintf('SRRC | BER=%.4f', BER_srrc));

sgtitle(sprintf('Full Image Recovery | SNR=%.1f dB | \\sigma=%.2f', 10*log10(1/sigma^2), sigma));

% figure;
% sgtitle(sprintf('Original vs Recovered Blocks — Pixel Space (N=%d, \\sigma=%.2f)', N, sigma));
% 
% for k = 1:N
%     % Original block: undo quantization -> IDCT
%     dct_orig = double(blocks(:,:,k)) / 255 * (max_value - min_value) + min_value;
%     pixel_orig = max(0, min(1, idct2(dct_orig)));
% 
%     % Half-Sine recovered block
%     dct_hs = double(blocks_rec_hs(:,:,k)) / 255 * (max_value - min_value) + min_value;
%     pixel_hs = max(0, min(1, idct2(dct_hs)));
% 
%     % SRRC recovered block
%     dct_srrc = double(blocks_rec_srrc(:,:,k)) / 255 * (max_value - min_value) + min_value;
%     pixel_srrc = max(0, min(1, idct2(dct_srrc)));
% 
%     % Row 1: Original
%     subplot(3, N, k);
%     imshow(pixel_orig, []);
%     if k == 1, ylabel('Original', 'FontWeight', 'bold'); end
%     title(sprintf('Block %d', k));
% 
%     % Row 2: Half-Sine recovered
%     subplot(3, N, N + k);
%     imshow(pixel_hs, []);
%     if k == 1, ylabel('Half-Sine', 'FontWeight', 'bold'); end
% 
%     % Compute per-block MSE for half-sine
%     mse_hs = mean((pixel_orig(:) - pixel_hs(:)).^2);
%     title(sprintf('MSE=%.4f', mse_hs));
% 
%     % Row 3: SRRC recovered
%     subplot(3, N, 2*N + k);
%     imshow(pixel_srrc, []);
%     if k == 1, ylabel('SRRC', 'FontWeight', 'bold'); end
% 
%     % Compute per-block MSE for SRRC
%     mse_srrc = mean((pixel_orig(:) - pixel_srrc(:)).^2);
%     title(sprintf('MSE=%.4f', mse_srrc));
% end