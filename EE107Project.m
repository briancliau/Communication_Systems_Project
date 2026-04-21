% % load image as floating point numbers
% img = imread("grayscale_cat.jpg");
% image = im2double(img);
% 
% % ensure divisbility by 8
% [m, n] = size(image);
% m = m - mod(m, 8);
% n = n - mod(n, 8);
% image = image(1:m, 1:n);
% 
% % apply 8x8 DCT blocks and create a 3D array of values
% fun = @(block_struct) dct2(block_struct.data);
% B = blockproc(image, [8 8], fun);
% 
% % Scale B
% min_value = min(B(:));
% max_value = max(B(:));
% B_scaled = (B - min_value) / (max_value - min_value);
% 
% % quantized B
% B_quantized = uint8(round(B_scaled * 255));
% 
% % Create 3D array
% blocks = reshape(B_quantized, [8, m/8, 8, n/8]);
% blocks = permute(blocks, [1, 3, 2, 4]);
% blocks = reshape (blocks, [8, 8, m*n/64]);
% 
% num_blocks = m  * n / 64;
% B_3d = reshape(B, 8, 8, num_blocks);
% 
% % N is the group size being transferred
% N = 5;
% 
% % Bit stream creation
% group = blocks(:, :, N);
% col_vector = double(group(:));
% bit_matrix = int2bit(col_vector, 8);
% bit_vector = bit_matrix(:)';
% 
% % Modulation
% symbols = 2*bit_vector - 1;

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
N_fft = 1024;  % use more points for smoother frequency plot
f = linspace(-Ns/2, Ns/2, N_fft);

% compute FFTs
g1_f = fftshift(fft(g1, N_fft));
g2_f = fftshift(fft(g2, N_fft));

% % --- Half-sine pulse ---
% figure;
% subplot(3,1,1);
% plot(linspace(0, 1, Ns), g1);   % time axis 0 to T=1
% xlabel('Time (t/T)');
% ylabel('Amplitude');
% title('Half-Sine Pulse - Time Domain');
% 
% subplot(3,1,2);
% plot(f, 20*log10(abs(g1_f)));   % magnitude in dB
% xlabel('Normalized Frequency (f*T)');
% ylabel('Magnitude (dB)');
% title('Half-Sine Pulse - Frequency Domain');
% 
% subplot(3,1,3);
% plot(f, rad2deg(angle(g1_f)));   % phase in degrees
% xlabel('Frequency (Hz)');
% ylabel('Phase (degrees)');
% title('Half-Sine Pulse - Frequency Domain Phase');
% % --- SRRC pulse ---
% figure;
% subplot(3,1,1);
% t_srrc = linspace(-K, K, length(g2));  % time axis spans -K to K
% plot(t_srrc, g2);
% xlabel('Time (t/T)');
% ylabel('Amplitude');
% title('SRRC Pulse - Time Domain');
% 
% subplot(3,1,2);
% plot(f, 20*log10(abs(g2_f)));   % magnitude in dB
% xlabel('Frequency (Hz)');
% ylabel('Magnitude (dB)');
% title('SRRC Pulse - Frequency Domain');
% 
% subplot(3,1,3);
% plot(f, rad2deg(angle(g2_f)));   % phase in degrees
% xlabel('Normalized Frequency (f*T)');
% ylabel('Phase (degrees)');
% title('Half-Sine Pulse - Frequency Domain Phase');


clf;
randbitstream = 2*randi([0 1], 1, 10)-1;
t_bits = 0:length(randbitstream)-1;
%stairs(t_bits, randbitstream)
%hold on
randbitstream = upsample(randbitstream,32);
randbitstream = circshift(randbitstream,16);
halfsinebits = conv(randbitstream,g1,'same');
SRRCbits = conv(randbitstream,g2,'same');
t_total1 = linspace(0, 10, length(randbitstream));
% plot(t_total1,halfsinebits)
% plot(t_total1,SRRCbits)
% figure;
% subplot(2,1,1);
% plot(f, 20*log10(abs(fftshift(fft(halfsinebits, N_fft)))));   % magnitude in dB
% xlabel('Frequency (Hz)');
% ylabel('Magnitude (dB)');
% title('Half-Sine Modulation - Frequency Domain');


% subplot(2,1,2);
% plot(f, 20*log10(abs(fftshift(fft(SRRCbits, N_fft)))));   % magnitude in dB
% xlabel('Frequency (Hz)');
% ylabel('Magnitude (dB)');
% title('SRRC Modulation - Frequency Domain');
% eyediagram(SRRCbits(32/2+1:end),32)
% clf;
h = [1,1/2,3/4,-2/7];
%freqz(h)
heff = [upsample(h, 32),zeros(1,320-128)];
L = 2^15;
channelhalfsine = filter(heff,1,halfsinebits);
channelSRRC = filter(heff,1,SRRCbits);
 %plot(t_total1,channelhalfsine)
 %hold on
 %plot(t_total1,channelSRRC(1:320))
 noise = .01*randn([1,length(channelSRRC)]);
 channelhalfsine = channelhalfsine+noise;
 channelSRRC = channelSRRC+noise;
 %plot(t_total, channelhalfsine+noise)
match_g2 = flip(g2);
channelSRRC = conv(channelSRRC, match_g2, 'same');
ones = [1, zeros(1, L-1)];
invertedchannel = filter(1,heff,ones);
channelSRRC = filter(invertedchannel,1,channelSRRC);
channelhalfsine = filter(invertedchannel,1,channelhalfsine);
eyediagram(channelSRRC(32/2+1:end),32)
%plot(invertedchannel)
%freqz(invertedchannel)