def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """提取音頻特徵
        
        Args:
            audio_path: 音頻文件路徑
            
        Returns:
            特徵字典，包含以下特徵：
            - mfcc: MFCC特徵 (n_frames, 13)
            - f0: 基頻特徵 (n_frames, 1)
            - energy: 能量特徵 (n_frames, 1)
            - zcr: 過零率特徵 (n_frames, 1)
        """
        try:
            # 載入音頻
            y, sr = self.load_audio(audio_path)
            
            # 計算幀長和幀移（以秒為單位）
            frame_length = 0.025  # 25ms
            frame_shift = 0.010   # 10ms
            
            # 將秒轉換為採樣點數
            frame_length_samples = int(frame_length * sr)
            frame_shift_samples = int(frame_shift * sr)
            
            # 計算總幀數
            n_frames = 1 + (len(y) - frame_length_samples) // frame_shift_samples
            
            # 初始化特徵數組
            mfcc_features = np.zeros((n_frames, 13))
            f0_features = np.zeros((n_frames, 1))
            energy_features = np.zeros((n_frames, 1))
            zcr_features = np.zeros((n_frames, 1))
            
            # 逐幀提取特徵
            for i in range(n_frames):
                # 提取當前幀
                start = i * frame_shift_samples
                end = start + frame_length_samples
                frame = y[start:end]
                
                # 提取MFCC
                mfcc = librosa.feature.mfcc(
                    y=frame,
                    sr=sr,
                    n_mfcc=13,
                    n_fft=frame_length_samples,
                    hop_length=frame_length_samples
                )
                mfcc_features[i] = mfcc[:, 0]
                
                # 提取F0
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    frame,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr,
                    frame_length=frame_length_samples,
                    hop_length=frame_length_samples
                )
                f0_features[i] = f0[0] if not np.isnan(f0[0]) else 0
                
                # 提取能量
                energy = librosa.feature.rms(
                    y=frame,
                    frame_length=frame_length_samples,
                    hop_length=frame_length_samples
                )
                energy_features[i] = energy[0]
                
                # 提取過零率
                zcr = librosa.feature.zero_crossing_rate(
                    frame,
                    frame_length=frame_length_samples,
                    hop_length=frame_length_samples
                )
                zcr_features[i] = zcr[0]
            
            # 正規化特徵
            mfcc_features = (mfcc_features - np.mean(mfcc_features, axis=0)) / (np.std(mfcc_features, axis=0) + 1e-8)
            f0_features = (f0_features - np.mean(f0_features)) / (np.std(f0_features) + 1e-8)
            energy_features = (energy_features - np.mean(energy_features)) / (np.std(energy_features) + 1e-8)
            zcr_features = (zcr_features - np.mean(zcr_features)) / (np.std(zcr_features) + 1e-8)
            
            return {
                'mfcc': mfcc_features,
                'f0': f0_features,
                'energy': energy_features,
                'zcr': zcr_features
            }
            
        except Exception as e:
            logger.error(f"特徵提取失敗: {str(e)}")
            raise 