import os
import numpy as np
import torch
import torchaudio


def resample(y, orig_sr, target_sr):
    if isinstance(y, np.ndarray):
        orig_dtype = y.dtype
        y = torch.from_numpy(y)
    else:
        orig_dtype = np.float32
    if y.dim() == 1:
        y = y.unsqueeze(0)
    y = y.float()
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    result = resampler(y)
    if result.size(0) == 1:
        result = result.squeeze(0)
    return result.numpy().astype(orig_dtype)


def load(path, sr=None):
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.numpy()
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    elif waveform.ndim == 2:
        waveform = waveform.mean(axis=0)
    if sr is not None and sr != sample_rate:
        waveform = resample(waveform, sample_rate, sr)
        sample_rate = sr
    return waveform, sample_rate


def normalize(S, norm=None, axis=None):
    if norm is None:
        return S
    if isinstance(S, np.ndarray):
        S = torch.from_numpy(S)
    if axis is None:
        axis = -1 if S.dim() > 1 else None
    if axis is not None:
        if norm == 'l2':
            norm_val = torch.norm(S, p=2, dim=axis, keepdim=True)
            norm_val[norm_val == 0] = 1.0
            S = S / norm_val
        elif norm == 'l1':
            norm_val = torch.norm(S, p=1, dim=axis, keepdim=True)
            norm_val[norm_val == 0] = 1.0
            S = S / norm_val
        elif isinstance(norm, (int, float)):
            max_val = torch.max(torch.abs(S), dim=axis, keepdim=True).values
            max_val[max_val == 0] = 1.0
            S = S / max_val * norm
    else:
        if norm == 'l2':
            S = S / torch.norm(S)
        elif norm == 'l1':
            S = S / torch.sum(torch.abs(S))
        elif isinstance(norm, (int, float)):
            S = S / torch.max(torch.abs(S)) * norm
    return S.numpy()


def find_files(path, ext=None):
    if ext is None:
        ext = 'wav'
    if not isinstance(ext, (list, tuple)):
        ext = [ext]
    ext = [e.lower() for e in ext]
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if any(filename.lower().endswith('.' + e) for e in ext):
                files.append(os.path.join(root, filename))
    return sorted(files)


def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=None, pad_mode='reflect'):
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    window_func = torch.hann_window(win_length, dtype=y.dtype, device=y.device)
    
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_func,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        return_complex=True
    )
    
    if spec.size(0) == 1:
        spec = spec.squeeze(0)
    
    return spec.numpy()


def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False):
    if fmax is None:
        fmax = sr / 2
    
    mel_filter = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=fmin,
        f_max=fmax,
        n_mels=n_mels,
        sample_rate=sr,
        norm='slaney' if not htk else None,
        mel_scale='htk' if htk else 'slaney'
    )
    
    return mel_filter.t().numpy()


class util:
    normalize = staticmethod(normalize)
    find_files = staticmethod(find_files)


class core:
    resample = staticmethod(resample)
    load = staticmethod(load)
    stft = staticmethod(stft)


class filters:
    mel = staticmethod(mel)