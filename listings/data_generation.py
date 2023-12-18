def generate_training_data(element_lines
, mu_max_err=0.002
, sigma_max_err=0.08
, mu_max_err_global=0.05
, samples=10
, elements_per_sample=3
, ...):
    Parameters:
        - element_lines (dict): Dictionary of element lines.
        - mu_max_err (float): Maximum error for mu of single peak.
        - sigma_max_err (float): Maximum error for sigma of single peak.
        - mu_max_err_global (float): Maximum error for mu of all peaks.
        - samples (int): Number of samples to generate.
        - elements_per_sample (int): Number of different elements per sample.

    Returns:
        - X (list): List of input data samples.
        - y (list): List of target data samples.
