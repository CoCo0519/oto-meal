Wavelet Tool Added

  - Created ppg_wavelet_denoise.py: standalone wavelet-approximation denoiser derived from the existing PPG analysis flow. It reads parameters from JSON, runs PyWavelets, and saves three PNGs (original,
  denoised, comparison with residual).
  - Added wavelet_denoise_config.json with editable knobs: sampling rate, channel, wavelet family, decomposition level, reconstruction mode, threshold strategy (universal/manual), scale/manual value, and
  plotting size.

  How To Run

  1. Ensure PyWavelets is installed: pip install PyWavelets.
  2. Edit wavelet_denoise_config.json to tune parameters.
  3. Example command:

     python ppg_wavelet_denoise.py -i .\hyx_data\sample.txt -j wavelet_denoise_config.json
     Options: --channel, --output-dir, --prefix, --fs override config defaults.

  The script writes outputs to <input>_wavelet/ (or your chosen directory) and reports the effective threshold and SNR gain. Natural follow-ups: tweak config parameters for different recordings or integrate
  the script into your processing pipeline.

▌ pip install PyWavelets

> I can’t run pip install here—the environment is read-only and has restricted network access. Please install PyWavelets directly in your own terminal:

  pip install PyWavelets

  Once that finishes, re-run the wavelet script and it’ll find the module.