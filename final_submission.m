% load image as floating point numbers
img = imread("mushroom.png");
if size(img, 3) == 3
    img = rgb2gray(img);
end
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

N = (n*m)/64;

figure;
sgtitle(sprintf('Original Image Blocks Passed Through Pipeline (N=%d blocks)', N));

for k = 1:N
    dct_block_scaled = double(blocks(:,:,k)) / 255 * (max_value - min_value) + min_value;
    pixel_block = idct2(dct_block_scaled);
    pixel_block = max(0, min(1, pixel_block));
    % fprintf('Block on: %d\n', k);
end

figure;
imshow(image, []);
title('Full Image — Highlighted Transmitted Blocks');
hold on;

% The bolcks are indexed in column-major order across the image.
% Block inedx k corresponds to rwo r, col c in the block grid:
num_block_cols = n / 8;
for k = 1:N
    block_col = ceil(k / (m/8));
    block_row = mod(k-1, m/8) + 1;
    px_x = (block_col - 1) * 8 + 1;
    px_y = (block_row - 1) * 8 + 1;
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

% Find -3dB bandwidth
mag1 = abs(g1_f);
mag2 = abs(g2_f);
peak1 = max(mag1);
peak2 = max(mag2);

bw1 = f(find(mag1 >= peak1/sqrt(2), 1, 'last'));
bw2 = f(find(mag2 >= peak2/sqrt(2), 1, 'last'));

fprintf('Half-Sine -3dB bandwidth: %.4f (normalized)\n', bw1);
fprintf('SRRC -3dB bandwidth: %.4f (normalized)\n', bw2);

clf;
numberofbits = length(symbols);
randbitstream = symbols;

randbitstream = upsample(randbitstream, 32);
%randbitstream = circshift(randbitstream, 15);
halfsinebits = conv(randbitstream, g1);
SRRCbits     = conv(randbitstream, g2);
t_total1 = linspace(0, 10, length(randbitstream));

h = [1,1/2,3/4,-2/7];
%h = [1,.4365,.1905,.0832,0,.0158,0,.003];
% h = [.5,1,0,.63,0,0,0,0,.25,0,0,0,.16,zeros(1,12),.1];
powergainin = norm(h)^2;
heff = [upsample(h, 32), zeros(1, numberofbits*32 - length(h)*32)];
L = 2^15;
channelhalfsine = conv(heff,halfsinebits);
channelSRRC = conv(heff,SRRCbits);
sigma = 1.5;
noise = sigma*randn([1,length(channelhalfsine)]);
channelhalfsine = channelhalfsine+noise;
noise = sigma*randn([1,length(channelSRRC)]);
channelSRRC = channelSRRC+noise;
match_g1 = flip(g1);
channelhalfsine = filter(match_g1,1,channelhalfsine);
match_g2 = flip(g2);
channelSRRC = filter(match_g2,1,channelSRRC);
ones = [1, zeros(1, L-1)];
invertedchannel = 1./fft(heff,length(channelSRRC));
%channelSRRC = ifft(fft(channelSRRC).*invertedchannel);
invertedchannel = 1./fft(heff,length(channelhalfsine));
%channelhalfsine = ifft(fft(channelhalfsine).*invertedchannel);
H = fft(heff,length(channelhalfsine));
MMSE = conj(H)./(abs(H).^2+sigma^2);
channelhalfsine = ifft(MMSE.*fft(channelhalfsine));

H = fft(heff,length(channelSRRC));
MMSE = conj(H)./(abs(H).^2+sigma^2);
channelSRRC = ifft(MMSE.*fft(channelSRRC));
% receivedhalfsine = channelhalfsine(1:numberofbits*32);
% receivedSRRC = channelSRRC(1:numberofbits*32);
receivedhalfsine = channelhalfsine;
receivedSRRC = channelSRRC;

sig_len = length(channelhalfsine);
H_full = fft(heff, sig_len);
MMSE_full = conj(H_full) ./ (abs(H_full).^2 + sigma^2);
mmse_imp = real(ifft(MMSE_full));  % MMSE impulse response

% Pad test impulse to same length
test_impulse = zeros(1, sig_len);
test_impulse(1) = 1;


% Half-sine: pulse -> matched filter -> MMSE
hs_tx = conv(test_impulse, g1);
hs_mf = conv(hs_tx(1:sig_len), match_g1);
hs_eq = conv(hs_mf(1:sig_len), mmse_imp);
[~, peak_hs] = max(abs(hs_eq(1 : sig_len)));
delay_hs = peak_hs - 1;

sig_len = length(channelSRRC);
H_full = fft(heff, sig_len);
MMSE_full = conj(H_full) ./ (abs(H_full).^2 + sigma^2);
mmse_imp = real(ifft(MMSE_full));

srrc_tx = conv(test_impulse, g2);
srrc_mf = conv(srrc_tx(1:sig_len), match_g2);
srrc_eq = conv(srrc_mf(1:sig_len), mmse_imp);
[~, peak_srrc] = max(abs(srrc_eq(1 : sig_len)));
delay_srrc = peak_srrc - 1;

idx_hs   = delay_hs   + (0:numberofbits-1)*Ns + 1;
idx_srrc = delay_srrc + (0:numberofbits-1)*Ns + 1;

if any(idx_hs > length(receivedhalfsine))
    warning('Half-sine: %d indices exceed signal length', sum(idx_hs > length(receivedhalfsine)));
    idx_hs = idx_hs(idx_hs <= length(receivedhalfsine));
end
if any(idx_srrc > length(receivedSRRC))
    warning('SRRC: %d indices exceed signal length', sum(idx_srrc > length(receivedSRRC)));
    idx_srrc = idx_srrc(idx_srrc <= length(receivedSRRC));
end

% Sample
sampled_hs = real(receivedhalfsine(idx_hs));
sampled_srrc = real(receivedSRRC(idx_srrc));

% Threshold detection
detected_hs = double(sampled_hs   > 0);
detected_srrc = double(sampled_srrc > 0);

% Pad to numberofbits if any indices were trimmed
detected_hs(end+1 : numberofbits) = 0;
detected_srrc(end+1 : numberofbits) = 0;

% Conversion to image

bit_mat_hs = reshape(detected_hs, 8, [])';
bit_mat_srrc = reshape(detected_srrc, 8, [])';

px_hs = uint8(bi2de(bit_mat_hs, 'left-msb'));
px_srrc = uint8(bi2de(bit_mat_srrc, 'left-msb'));

N_blocks_transmitted = N;
blocks_rec_hs = reshape(px_hs, [8, 8, N_blocks_transmitted]);
blocks_rec_srrc = reshape(px_srrc, [8, 8, N_blocks_transmitted]);

% Image post processing

B_rec_hs = double(blocks_rec_hs) / 255 * (max_value - min_value) + min_value;
B_rec_srrc = double(blocks_rec_srrc) / 255 * (max_value - min_value) + min_value;

B_rec_2d_hs = reshape(permute(reshape(B_rec_hs,   [8, 8, 1, N]), [1,3,2,4]), [8, N*8]);
B_rec_2d_srrc = reshape(permute(reshape(B_rec_srrc, [8, 8, 1, N]), [1,3,2,4]), [8, N*8]);

ifun = @(block_struct) idct2(block_struct.data);
img_out_hs = blockproc(B_rec_2d_hs, [8 8], ifun);
img_out_srrc = blockproc(B_rec_2d_srrc, [8 8], ifun);

img_out_hs = max(0, min(1, img_out_hs));
img_out_srrc = max(0, min(1, img_out_srrc));

B_rec_hs_full = double(blocks_rec_hs) / 255 * (max_value - min_value) + min_value;
B_rec_srrc_full = double(blocks_rec_srrc) / 255 * (max_value - min_value) + min_value;

B_rec_2d_hs_full = reshape(permute(reshape(B_rec_hs_full, [8, 8, m/8, n/8]), [1,3,2,4]), [m, n]);
B_rec_2d_srrc_full = reshape(permute(reshape(B_rec_srrc_full, [8, 8, m/8, n/8]), [1,3,2,4]), [m, n]);

ifun = @(block_struct) idct2(block_struct.data);
img_out_hs_full = blockproc(B_rec_2d_hs_full, [8 8], ifun);
img_out_srrc_full = blockproc(B_rec_2d_srrc_full, [8 8], ifun);

img_out_hs_full = max(0, min(1, img_out_hs_full));
img_out_srrc_full = max(0, min(1, img_out_srrc_full));

% BER
BER_hs= sum(detected_hs ~= double(bit_vector)) / numberofbits;
BER_srrc = sum(detected_srrc ~= double(bit_vector)) / numberofbits;
fprintf('SNR = %.2f dB\n', 10*log10(1/sigma^2));
fprintf('BER Half-Sine : %.6f\n', BER_hs);
fprintf('BER SRRC : %.6f\n', BER_srrc);

% Comparison figure
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

% p_hs = conv(g1, match_g1);
% p_srrc = conv(g2, match_g2);
% 
% figure;
% subplot(2,1,1);
% plot(p_hs);
% title('Half-Sine: g * matched filter (continuous)');
% xlabel('Samples'); ylabel('Amplitude');
% grid on;
% 
% subplot(2,1,2);
% plot(p_srrc);
% title('SRRC: g * matched filter (continuous)');
% xlabel('Samples'); ylabel('Amplitude');
% grid on;
% 
% sgtitle('Q17: Nyquist Zero-ISI Criterion Check (post matched filter)');
% 
% figure; hold on;
% K_values = [2, 3, 4, 6];
% colors = lines(length(K_values));
% 
% for i = 1:length(K_values)
%     K_test = K_values(i);
%     g_test = rcosdesign(alpha, 2*K_test, Ns);
%     g_test = g_test / sqrt(sum(g_test.^2) * 1/Ns);
%     G_test = fftshift(fft(g_test, N_fft));
%     plot(f, 20*log10(abs(G_test)), 'Color', colors(i,:), ...
%          'DisplayName', sprintf('K=%d', K_test));
% end
% 
% xlabel('Normalized Frequency'); ylabel('Magnitude (dB)');
% title('SRRC Bandwidth vs Truncation Length K');
% legend; grid on; xlim([-3 3]); ylim([-60 35]);
% 
% figure; hold on;
% alpha_values = [0.1, 0.3, 0.5, 0.9];
% 
% for i = 1:length(alpha_values)
%     g_test = rcosdesign(alpha_values(i), 2*K, Ns);
%     g_test = g_test / sqrt(sum(g_test.^2) * 1/Ns);
%     G_test = fftshift(fft(g_test, N_fft));
%     plot(f, 20*log10(abs(G_test)), ...
%          'DisplayName', sprintf('\\alpha=%.1f', alpha_values(i)));
% end
% 
% xlabel('Normalized Frequency'); ylabel('Magnitude (dB)');
% title('SRRC Bandwidth vs Roll-off Factor \alpha');
% legend; grid on; xlim([-3 3]); ylim([-60 35]);
% 
% figure;
% subplot(2,1,1);
% plot(f, 20*log10(abs(g1_f)));
% title('Half-Sine Frequency Response (dB)');
% xlabel('Normalized Frequency (×F_s)'); ylabel('Magnitude (dB)');
% grid on; xlim([-3 3]);
% 
% subplot(2,1,2);
% plot(f, 20*log10(abs(g2_f)));
% title('SRRC Frequency Response (dB)');
% xlabel('Normalized Frequency (×F_s)'); ylabel('Magnitude (dB)');
% grid on; xlim([-3 3]);
% 
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
