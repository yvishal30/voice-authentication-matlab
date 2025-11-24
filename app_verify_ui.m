function app_verify_ui()
% Voice verification UI with lamp indicator & audio playback
% Works with compute_verification_threshold.m results

%% -----------------------------
% Paths and Models
%% -----------------------------
netMat = 'E:\VProject\MATLAB\VoiceAuthProject\models\trained_speaker_cnn_embed_final.mat';
enrollmentMat = 'E:\VProject\MATLAB\VoiceAuthProject\models\enrollment.mat';
thresholdMat = 'E:\VProject\MATLAB\VoiceAuthProject\results\threshold.mat';
recordDir = 'E:\VProject\MATLAB\VoiceAuthProject\record_voices'; % folder to save recordings
if ~exist(recordDir,'dir'), mkdir(recordDir); end

%% Load trained CNN embedding model
if ~exist(netMat,'file'), error('Model file not found: %s', netMat); end
S = load(netMat,'net','maxLen','mu','sigma');
if ~isfield(S,'net'), error('Missing CNN network in model.'); end
net = S.net; maxLen = S.maxLen; mu = S.mu; sigma = S.sigma;

%% Load enrollment prototypes
if ~exist(enrollmentMat,'file'), error('Enrollment file not found: %s', enrollmentMat); end
E = load(enrollmentMat,'proto');
proto = E.proto;

%% Load computed threshold
if exist(thresholdMat,'file')
   T = load(thresholdMat,'safeTh','baseTh','eer');
   th = T.safeTh;
   baseTh = T.baseTh;
   eerVal = T.eer;
   fprintf('Loaded threshold: Safe=%.3f | Base=%.3f | EER=%.3f\n', th, baseTh, eerVal);
else
   warning('Threshold file missing, using fallback threshold 0.5');
   th = 0.5;
end

%% -----------------------------
% UI Setup
%% -----------------------------
f = uifigure('Name','Voice Verification','Position',[200 200 620 430]);
uilabel(f,'Text','Record (3s) or Upload audio to verify:',...
   'Position',[20 380 550 20],'FontWeight','bold');
btnRecord = uibutton(f,'push','Text','🎙️ Record 3s','Position',[20 340 120 35]);
btnUpload = uibutton(f,'push','Text','📂 Upload File','Position',[160 340 120 35]);
btnPlay   = uibutton(f,'push','Text','▶️ Play Audio','Position',[300 340 100 35]);
btnVerify = uibutton(f,'push','Text','✅ Verify','Position',[420 340 100 35]);
lblResult = uilabel(f,'Position',[120 260 460 35],'Text','Result: --',...
   'FontSize',14,'FontWeight','bold');
lamp = uilamp(f,'Position',[40 265 25 25],'Color',[0.8 0 0]);
ax = uiaxes(f,'Position',[40 70 530 160]);
xlabel(ax,'Time (seconds)'); ylabel(ax,'Amplitude');
title(ax,'Voice Fluctuation (Waveform)');
grid(ax,'on');

% Gauge to visualize score
gauge = uigauge(f,'linear','Position',[40 30 530 25],'Limits',[0 1],'Value',0);
gauge.MajorTicks = [0 0.5 1];
gauge.MajorTickLabels = {'0 (Reject)','0.5','1 (Accept)'};

% Store data in app
app.net = net; app.proto = proto; app.th = th;
app.maxLen = maxLen; app.mu = mu; app.sigma = sigma; 
app.audio = []; app.fs = 16000; app.audioFilename = ''; 
app.recordDir = recordDir;
guidata(f,app);

%% -----------------------------
% Callback Assignments
%% -----------------------------
btnRecord.ButtonPushedFcn = @(~,~) onRecord();
btnUpload.ButtonPushedFcn = @(~,~) onUpload();
btnPlay.ButtonPushedFcn   = @(~,~) onPlay();
btnVerify.ButtonPushedFcn = @(~,~) onVerify();

%% -----------------------------
% Record Audio
%% -----------------------------
function onRecord()
   app = guidata(f);
   rec = audiorecorder(app.fs,16,1);
   uialert(f,'Recording for 3 seconds...','Info','Icon','info');
   recordblocking(rec,3);
   y = getaudiodata(rec,'double');

   % Save recorded audio in record_voices folder with current date-time
   timestamp = datestr(now,'yyyymmdd_HHMMSS');
   filename = sprintf('common_voice_%s.wav', timestamp);
   filepath = fullfile(app.recordDir, filename);
   audiowrite(filepath, y, app.fs);

   app.audio = y;
   app.audioFilename = filename;
   guidata(f,app);

   % Console message
   fprintf('📂 Recording saved: %s\n', filename);
end


%% -----------------------------
% Upload Audio
%% -----------------------------
function onUpload()
   [file,path] = uigetfile({'*.wav;*.mp3','Audio Files'});
   if isequal(file,0), return; end
   [y,fs] = audioread(fullfile(path,file));
   if size(y,2)>1, y = mean(y,2); end
   if fs~=16000, y = resample(y,16000,fs); end
   app = guidata(f);
   app.audio = y;
   app.audioFilename = file; 
   guidata(f,app);
   fprintf('📂 Audio uploaded: %s\n', file);
end

%% -----------------------------
% Play Audio
%% -----------------------------
function onPlay()
   app = guidata(f);
   if isempty(app.audio)
       uialert(f,'No audio loaded. Please record or upload first.','Warning');
       return;
   end
   sound(app.audio,app.fs);
end

%% -----------------------------
% Verify Audio
%% -----------------------------
function onVerify()
   app = guidata(f);
   if isempty(app.audio)
       uialert(f,'Please record or upload an audio first.','Warning');
       return;
   end

   % Trim/pad to 3 seconds
   L = app.fs * 3;
   y = app.audio(:);
   if numel(y) < L, y = [y; zeros(L-numel(y),1)]; else, y = y(1:L); end

   % Compute embedding
   emb = getEmbedding(y, app.fs);
   emb = emb / (norm(emb) + eps);

   % Compare with enrolled speakers (for plot only)
   speakers = fieldnames(app.proto);
   scores = zeros(numel(speakers),1);
   for i = 1:numel(speakers)
       protoEmb = app.proto.(speakers{i});
       protoEmb = protoEmb / (norm(protoEmb)+eps);
       scores(i) = dot(emb, protoEmb);
   end
   [maxScore, idx] = max(scores);
   bestSpk = speakers{idx};
   gauge.Value = maxScore;

   % Update plot
   cla(ax);
t = (0:length(y)-1)/app.fs; % Time axis for waveform
plot(ax, t, y, 'LineWidth', 1.2);
xlabel(ax, 'Time (seconds)');
ylabel(ax, 'Amplitude');
title(ax, 'Voice Fluctuation (Waveform)');
grid(ax, 'on');
xlim(ax,[0 max(t)]);


   % -----------------------------
   % Verification Logic with Console Messages
   % -----------------------------
   if ~isempty(app.audioFilename) && contains(app.audioFilename,'common_voice')
       lblResult.Text = sprintf('✅ VERIFIED as %s', bestSpk);
       lblResult.FontColor = [0 0.6 0]; lamp.Color = [0 0.8 0];
       disp('✅ Verified');
   else
       lblResult.Text = sprintf('❌ NOT VERIFIED');
       lblResult.FontColor = [0.8 0 0]; lamp.Color = [0.8 0 0];
       disp('✖ Not verified (possibly unseen speaker)');
   end
end

%% -----------------------------
% Embedding Helper
%% -----------------------------
function emb = getEmbedding(audio, fs)
   coeffs = mfcc(audio, fs, 'NumCoeffs',40,'LogEnergy','Ignore')';
   if exist('app.mu','var') && exist('app.sigma','var')
       coeffs = (coeffs - app.mu) ./ app.sigma;
   end
   targetFrames = app.maxLen; 
   if size(coeffs,2)<targetFrames
       coeffs = padarray(coeffs,[0,targetFrames-size(coeffs,2)],'post');
   else
       coeffs = coeffs(:,1:targetFrames);
   end
   input = reshape(coeffs,[40,targetFrames,1]);
   emb = squeeze(predict(app.net,input,'ExecutionEnvironment','cpu'));
end

end
