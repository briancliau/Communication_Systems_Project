%% Image Processing
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

% N is the group size being transferred
N = 5;

%% Bit Stream Creation
group = blocks(:, :, 1:N);
col_vector = double(group(:));
bit_matrix = int2bit(col_vector, 8);
bit_vector = bit_matrix(:)';

%% Modulation
symbols = 2*bit_vector - 1;

% Half-sine pulsing shape function
Ns = 32;
t = linspace(0, 1, Ns+1);
t = t(1:Ns);
Tb = 1;
dt = Tb / Ns;
g1 = sin(pi*t);
A1 = 1 / sqrt(sum(g1.^2) * dt);
g1 = A1 * g1;

% SRRC pulsing shape function
alpha = 1;
K = 2;
g2 = rcosdesign(alpha, 2*K, Ns);
A2 = 1 / sqrt(sum(g2.^2) * dt);
g2 = A2 * g2;


% frequency axis (-0.5 to 0.5 normalized frequency)
N_fft = 1024;
fs = Ns / Tb;                       
f = linspace(-fs/2, fs/2, N_fft);   

% compute FFTs
g1_f = fftshift(fft(g1, N_fft));
g2_f = fftshift(fft(g2, N_fft));

g1_plot = g1;
g1_plot(end + 1) = 0;

% Half-sine pulse
figure;
subplot(3,1,1);
plot(linspace(0, 1, Ns+1), g1_plot);   % time axis 0 to T=1
xlabel('Time (t/T)');
ylabel('Amplitude');
title('(a) Half-Sine Pulse - Time Domain');

subplot(3,1,2);
plot(f, 20*log10(abs(g1_f)));   % magnitude in dB
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('(b) Half-Sine Pulse - Frequency Domain');

subplot(3,1,3);
plot(f, rad2deg(angle(g1_f)));
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('(c) Half-Sine Pulse - Phase Response');


% SRRC pulse
figure;
subplot(3,1,1);
t_srrc = linspace(-K, K, 2 * K * Ns + 1);  % time axis spans -K to K
plot(t_srrc, g2);
xlabel('Time (t/T)');
ylabel('Amplitude');
title('(a) SRRC Pulse - Time Domain');

subplot(3,1,2);
plot(f, 20*log10(abs(g2_f)));   % magnitude in dB
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('(b) SRRC Pulse - Frequency Domain');

subplot(3,1,3);
plot(f, rad2deg(angle(g2_f)));
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('(c) SRRC Pulse - Phase Response');

% Bandwidth for g1
power_g1 = abs(g1_f).^2;
total_power_g1 = sum(power_g1);
cumulative_power_g1 = cumsum(power_g1) / total_power_g1;

low_idx = find(cumulative_power_g1 >= 0.005, 1, 'first');
high_idx = find(cumulative_power_g1 >= 0.995, 1, 'first');

f_low_g1 = f(low_idx);
f_high_g1 = f(high_idx);
bandwidth_99_g1 = f_high_g1 - f_low_g1

% Bandwidth for g2
power_g2 = abs(g2_f).^2;
total_power_g2 = sum(power_g2);
cumulative_power_g2 = cumsum(power_g2) / total_power_g2;

low_idx = find(cumulative_power_g2 >= 0.005, 1, 'first');
high_idx = find(cumulative_power_g2 >= 0.995, 1, 'first');

f_low_g2 = f(low_idx);
f_high_g2 = f(high_idx);
bandwidth_99_g2 = f_high_g2 - f_low_g2

%% Example Bit Stream & Pulse Shaping
clf;
randbitstream = 2*randi([0 1], 1, 10)-1;
t_bits = 0:length(randbitstream)-1;
bit_stream = randbitstream;
stairs(t_bits, randbitstream)
hold on
randbitstream = upsample(randbitstream,32);
randbitstream = circshift(randbitstream,16);
halfsinebits = conv(randbitstream,g1,'same');
SRRCbits = conv(randbitstream,g2,'same');
t_total1 = linspace(0, 10, length(randbitstream));
plot(t_total1,halfsinebits)
plot(t_total1,SRRCbits)
figure;
subplot(2,1,1);
plot(f, 20*log10(abs(fftshift(fft(halfsinebits, N_fft)))));   % magnitude in dB
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Half-Sine Modulation - Frequency Domain');


subplot(2,1,2);
plot(f, 20*log10(abs(fftshift(fft(SRRCbits, N_fft)))));   % magnitude in dB
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('SRRC Modulation - Frequency Domain');

eyediagram(halfsinebits(Ns/2+1:end),32)
title("Half-Sine Eye Diagram");

%% Channel
h = [1, 1/2, 3/4, -2/7]; 
h_up = upsample(h, Ns);

% pass modulated signals through channel
s1_channel = conv(halfsinebits, h_up, 'same');
s2_channel = conv(SRRCbits, h_up, 'same');

% plot channel impulse and frequency response (use original h before upsampling)
H = fftshift(fft(h, N_fft));

figure;
subplot(3,1,1);
stem(h);                              % impulse response
xlabel('Tap');
ylabel('Amplitude');
title('(a) Channel - Impulse Response');

subplot(3,1,2);
plot(f, 20*log10(abs(H)));
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('(b) Channel - Frequency Response');

subplot(3,1,3);
plot(f, rad2deg(angle(H)));
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('(c) Channel - Phase Response');

figure
stairs(t_bits, bit_stream);
hold on;
plot(t_total1, s1_channel);
plot(t_total1, s2_channel);
legend("Bit Stream", "S1", "S2");
hold off;

eyediagram(s1_channel(16+1:end), Ns);
title('Half-Sine Eye Diagram After Channel');

eyediagram(s2_channel, Ns);
title('Eye Diagram');

%% Gaussian Noise
sigma_squared = 0.05;
sigma = sqrt(sigma_squared);
noise = sigma*randn(size(s1_channel), like=s1_channel);

% Add Gaussian noise to the channel output
s1_noisy = s1_channel + noise;
s2_noisy = s2_channel + noise;

figure
stairs(t_bits, bit_stream);
hold on;
plot(t_total1, s1_noisy);
plot(t_total1, s2_noisy);
legend("Bit Stream", "S1", "S2");
hold off;

eyediagram(s1_noisy(16+1:end), Ns);
title('Half-Sine Eye Diagram After Channel and Noise');

eyediagram(s2_noisy, Ns);
title('SRRC Eye Diagram After Channel and Noise');