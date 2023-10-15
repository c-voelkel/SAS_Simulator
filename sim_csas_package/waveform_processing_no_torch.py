#import torch
import math
import numpy as np

def hilbert_torch(x):
    N = len(x)                                                                  
                                                                                
    # Take forward fourier transform                                            
    Xf = np.fft.fft(x) #torch.fft.fft(x)
    h = np.zeros(N) #torch.zeros((N))
                                                                                
    if N % 2 == 0:                                                              
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:                                                                       
        h[0] = 1
        h[1:(N + 1) // 2] = 2
                                                                                
    # Take inverse Fourier transform                                            
    x_hilbert = np.fft.ifft(Xf * h) #torch.fft.ifft(Xf * h.to(Xf.device))

    return x_hilbert 


def replica_correlate_torch(x, kernel):
    #assert not x.dtype == torch.complex, "x should be real"
    # Forward fourier transform of received waveform
    x_hil = hilbert_torch(x)
    x_fft = np.fft.fft(x_hil)

    # Definition of cross-correlation
    x_rc = np.fft.ifft(x_fft * np.conj(kernel))
        
    return x_rc

# set min_dist to RP.min_dist if RP.generate_transmit_signal(crop_wfm=True)
def delay_waveforms(RP, ps, amplitudes, noise=False, noise_std=.01, temps=None,
    min_dist=0.0, scat_phase=None):

    amplitudes = amplitudes.squeeze().flatten()

    if scat_phase is not None:
      scat_p = scat_phase.squeeze().flatten()
    else:
      scat_p = np.zeros_like(amplitudes)#.to(amplitudes.device)
      
    #scat_p = scat_p.float()
    #ps_term = np.exp(np.complex128(real=np.array([0.0]), imag=-1*scat_p))
    ps_term = np.exp(np.complex128(0.0 - 1j * scat_p))

    # wfms = np.zeros((len(RP.trans), RP.num_samples),
    #     dtype=np.complex128).to(ps.device)
    #
    # df = RP.Fs / RP.num_samples
    # f_ind = np.linspace(0, int(RP.num_samples - 1), steps=int(RP.num_samples),
    #     dtype=np.float64)
    # f = f_ind * df
    # f[f > (RP.Fs / 2)] -= RP.Fs
    # w = (2 * math.pi * f).to(ps.device)
    #
    # # phase shift term
    # for index in range(len(RP.trans)):
    #     t1 = torch.sqrt(torch.sum((RP.trans[index].tx_pos[None, ...]\
    #          - ps) ** 2, 1))
    #     t2 = torch.sqrt(torch.sum((RP.trans[index].rx_pos[None, ...]\
    #          - ps) ** 2, 1))
    #
    #     # If no provided per ping temperature measurements
    #     if not temps:
    #       # min_dist shifts waveform down since we crop the waveform to scene
    #       # size
    #       tau = ((t1 + t2) - min_dist) / RP.c
    #     else:
    #       # from physics
    #       c = 331.4 + 0.6*temps[index]
    #       tau = ((t1 + t2) - min_dist) / c
    #
    #     phase = tau[:, None] * w[None, :]
    #
    #     complex_phase = torch.complex(torch.zeros_like(phase).to(phase.device),
    #                     -1*phase)
    #
    #     pr = torch.exp(complex_phase)*ps_term[..., None]
    #
    #     tsd_fft = RP.pulse_fft_kernel[None, :] * pr
    #     tsd = torch.fft.ifft(tsd_fft, dim=1)
    #
    #     tsd_scaled = amplitudes[:, None] * tsd.real
    #
    #     if noise:
    #         ps_noise = torch.normal(0, noise_std, size=tsd_scaled.shape).to(ps.device)
    #         tsd_scaled = tsd_scaled + ps_noise
    #
    #     tsd_sum = torch.sum(tsd_scaled, 0)
    #
    #     wfms[index, ...] = replica_correlate_torch(tsd_sum, RP.pulse_fft_kernel)

    wfms = np.zeros((len(RP.trans), RP.num_samples), dtype=np.complex128)

    df = RP.Fs / RP.num_samples
    f_ind = np.linspace(0, int(RP.num_samples - 1), num=int(RP.num_samples), dtype=np.float64)
    f = f_ind * df
    f[f > (RP.Fs / 2)] -= RP.Fs
    w = 2 * math.pi * f

    for index in range(len(RP.trans)):
        print("index", index)
        t1 = np.sqrt(np.sum((RP.trans[index].tx_pos[None, ...] - ps) ** 2, axis=1))
        t2 = np.sqrt(np.sum((RP.trans[index].rx_pos[None, ...] - ps) ** 2, axis=1))

        #t1 = np.sqrt(np.sum((RP.trans[index].tx_pos[None, ...] - ps) ** 2, 1))
        #t2 = np.sqrt(np.sum((RP.trans[index].rx_pos[None, ...] - ps) ** 2, 1))

        if not temps:
            tau = ((t1 + t2) - min_dist) / RP.c
        else:
            c = 331.4 + 0.6 * temps[index]
            tau = ((t1 + t2) - min_dist) / c

        phase = tau[:, None] * w[None, :]

        complex_phase = np.exp(1j * -phase)

        pr = complex_phase * ps_term[..., None]

        tsd_fft = RP.pulse_fft_kernel[None, :] * pr
        tsd = np.fft.ifft(tsd_fft, axis=1)

        tsd_scaled = amplitudes[:, None] * tsd.real

        if noise:
            ps_noise = np.random.normal(0, noise_std, size=tsd_scaled.shape)
            tsd_scaled = tsd_scaled + ps_noise

        tsd_sum = np.sum(tsd_scaled, axis=0)

        wfms[index, ...] = replica_correlate_torch(tsd_sum, RP.pulse_fft_kernel)

    return wfms

    

