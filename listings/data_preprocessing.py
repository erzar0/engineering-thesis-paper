def get_processed_spectrum(e_xy, peaks, energies, min_energy=0, max_energy=30
                           , energy_margin=1, target_length=4096, points_to_extrapolate_count=100, ...):
                           
    Parameters:
        - e_xy: numpy array, shape (H, W, C), representing the raw spectrum.
        - peaks: array-like, containing peak positions for energy calibration.
        - energies: array-like, containing corresponding energies for calibration peaks.
        - min_energy: float, optional, default: 0, the minimum energy boundry.
        - max_energy: float, optional, default: 30, the maximum energy boundry.
        - energy_margin: float, optional, default: 1, margin to mask at the edges of the spectrum.
        - target_length: int, optional, default: 4096, the desired length of the output spectrum.
        - points_to_extrapolate_count: int, optional, default: 100, number of points used for extrapolation at boundries of spectrum.

    Returns:
        - channel_energies: numpy array, shape (target_length,), the interpolated channel energies.
        - spectrum_acc: numpy array, shape (target_length,), accumulated and processed spectrum.

