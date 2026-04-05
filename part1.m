% load image as floating point numbers
img = imread("grayscale_cat.jpg");
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
blocks = reshape (blocks, [8, 8, m*n/64]);

num_blocks = m  * n / 64;
B_3d = reshape(B, 8, 8, num_blocks);

% N is the group size being transferred
N = 5;

% Bit stream creation
group = blocks(:, :, N);
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
g1 = g1/norm(g1);

% SRRC pulsing shape function
alpha = 0.5;
K = 6;
g2 = rcosdesign(alpha, 2*K, Ns);
g2 = g2/norm(g2);


% frequency axis (-0.5 to 0.5 normalized frequency)
N_fft = 1024;  % use more points for smoother frequency plot
f = linspace(-0.5, 0.5, N_fft);

% compute FFTs
g1_f = fftshift(fft(g1, N_fft));
g2_f = fftshift(fft(g2, N_fft));

% --- Half-sine pulse ---
figure;
subplot(2,1,1);
plot(linspace(0, 1, Ns), g1);   % time axis 0 to T=1
xlabel('Time (t/T)');
ylabel('Amplitude');
title('Half-Sine Pulse - Time Domain');

subplot(2,1,2);
plot(f, 20*log10(abs(g1_f)));   % magnitude in dB
xlabel('Normalized Frequency (f*T)');
ylabel('Magnitude (dB)');
title('Half-Sine Pulse - Frequency Domain');

% --- SRRC pulse ---
figure;
subplot(2,1,1);
t_srrc = linspace(-K, K, 2*K*Ns);  % time axis spans -K to K
plot(t_srrc, g2);
xlabel('Time (t/T)');
ylabel('Amplitude');
title('SRRC Pulse - Time Domain');

subplot(2,1,2);
plot(f, 20*log10(abs(g2_f)));   % magnitude in dB
xlabel('Normalized Frequency (f*T)');
ylabel('Magnitude (dB)');
title('SRRC Pulse - Frequency Domain');