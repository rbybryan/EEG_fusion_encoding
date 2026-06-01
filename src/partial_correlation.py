"""
Partial Correlation Analysis for EEG Encoding Models

This script computes partial correlations between fusion predictions and biological EEG data,
controlling for individual modality predictions (e.g., controlling for text when evaluating 
fusion performance beyond text alone).

The partial correlation removes the linear relationship between the control variable and both
the predictor and outcome variables, then computes the correlation of the residuals.

Usage:
    python partial_correlation.py --sub 4 --fusion_file fusion_model --control_file control_model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
from sklearn.utils import resample


def partial_corr_lstsq(x, y, control):
    """
    Computes the partial correlation between x and y, controlling for 'control',
    using direct linear algebra (least squares).
    
    The partial correlation formula:
    1. Regress x on control: x_residuals = x - predict(x | control)
    2. Regress y on control: y_residuals = y - predict(y | control) 
    3. Compute correlation between residuals: corr(x_residuals, y_residuals)
    
    Parameters
    ----------
    x : array-like
        Predictor variable (e.g., fusion prediction)
    y : array-like  
        Outcome variable (e.g., true EEG)
    control : array-like
        Control variable (e.g., text prediction)
        
    Returns
    -------
    r : float
        Partial correlation coefficient
    """
    x = np.asarray(x)
    y = np.asarray(y)
    control = np.asarray(control)
    
    # Add constant column for the intercept
    ones = np.ones((control.shape[0], 1))
    X = np.hstack([ones, control.reshape(-1, 1)])  # shape: (n_samples, 2)

    # Solve for x-residuals: x_residuals = x - X @ beta_x
    beta_x, _, _, _ = np.linalg.lstsq(X, x, rcond=None)
    resid_x = x - X @ beta_x

    # Solve for y-residuals: y_residuals = y - X @ beta_y
    beta_y, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid_y = y - X @ beta_y

    # Compute correlation of residuals
    r = np.corrcoef(resid_x, resid_y)[0, 1]
    return r


def plot_partial_correlation(fusion_test, control_test, bio_test, ch_names, times, 
                           control_name, sub, data_path, iterations=100):
    """
    Compute partial correlations between fusion and biological EEG data,
    controlling for a specific modality (e.g., text or image).

    Parameters
    ----------
    fusion_test : ndarray
        Fusion model predictions (n_samples × n_channels × n_timepoints).
    control_test : ndarray
        Control modality predictions (n_samples × n_channels × n_timepoints).
    bio_test : ndarray
        Biological EEG data (n_samples × n_repetitions × n_channels × n_timepoints).
    ch_names : list
        List of EEG channel names.
    times : ndarray
        EEG time points.
    control_name : str
        Name of control modality (e.g., 'text', 'image').
    sub : int
        Subject identifier.
    data_path : str
        Path to save the correlation results.
    iterations : int, default=100
        Number of bootstrap iterations for noise ceiling estimation.
    """
    
    # Ensure correct data shapes
    if len(fusion_test.shape) == 3:
        fusion_test = fusion_test[:, :63, :]  # Limit to first 63 channels
    else:   
        fusion_test = fusion_test.reshape(200, -1, 100)[:, :63, :]
    
    if len(control_test.shape) == 3:
        control_test = control_test[:, :63, :]
    else:
        control_test = control_test.reshape(200, -1, 100)[:, :63, :]
        
    bio_test = bio_test[:, :, :63, :]  # Limit to first 63 channels

    print(f"Data shapes - Fusion: {fusion_test.shape}, Control: {control_test.shape}, Bio: {bio_test.shape}")

    # Initialize result matrices
    n_channels, n_timepoints = bio_test.shape[2], bio_test.shape[3]
    partial_correlation_end = np.zeros((iterations, n_channels, n_timepoints))

    # Loop over bootstrap iterations
    for i in tqdm.tqdm(range(iterations), desc=f"Computing partial correlations (controlling for {control_name})"):
        # Random data repetitions index for train/test split
        shuffle_idx = resample(np.arange(bio_test.shape[1]), replace=False)[:bio_test.shape[1] // 2]
        
        # Average across one half of the biological data repetitions
        bio_data_avg_half_1 = np.mean(np.delete(bio_test, shuffle_idx, axis=1), axis=1)

        # Loop over EEG time points and channels
        for t in range(n_timepoints):
            # Skip time points with no predictions (zeros)
            if np.sum(fusion_test[:, :, t]) != 0:
                for c in range(n_channels):
                    try:
                        # Compute partial correlation: fusion vs bio, controlling for control modality
                        partial_correlation_end[i, c, t] = partial_corr_lstsq(
                            fusion_test[:, c, t],      # Predictor (fusion)
                            bio_data_avg_half_1[:, c, t],  # Outcome (true EEG)
                            control_test[:, c, t]      # Control variable
                        )
                    except (np.linalg.LinAlgError, ValueError):
                        # Handle singular matrices or invalid correlations
                        partial_correlation_end[i, c, t] = 0.0

    # Average the results across iterations
    partial_correlation = np.nanmean(partial_correlation_end, axis=0)
    
    print(f"Partial correlation range: [{np.nanmin(partial_correlation):.3f}, {np.nanmax(partial_correlation):.3f}]")
    print(f"Mean partial correlation: {np.nanmean(partial_correlation):.3f}")

    # Save results
    results_dict = {
        'partial_correlation': partial_correlation,
        'control_modality': control_name,
        'times': times,
        'ch_names': ch_names
    }

    save_dir = os.path.join(data_path, f'sub-{sub:02d}', 'partial_correlation')
    os.makedirs(save_dir, exist_ok=True)
    
    file_name = f'partial_correlation_fusion_controlling_{control_name}.npy'
    np.save(os.path.join(save_dir, file_name), results_dict)
    
    print(f"Results saved to: {os.path.join(save_dir, file_name)}")

    # Optional plotting
    plt.figure(figsize=(10, 6))
    plt.plot(times, np.nanmean(partial_correlation, axis=0), 
             label=f'Fusion (controlling for {control_name})', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Partial Correlation")
    plt.title(f'Subject {sub:02d} - Partial Correlation Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(save_dir, f'partial_correlation_fusion_controlling_{control_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return partial_correlation


def load_eeg_data(project_dir, sub):
    """Load biological EEG test data."""
    bio_file = os.path.join(project_dir, 'eeg_dataset', 'preprocessed_eeg_data_v1', 
                           f'eeg_sub-{sub:02d}_split-test.npy')
    
    if not os.path.exists(bio_file):
        raise FileNotFoundError(f"Biological EEG file not found: {bio_file}")
    
    bio_data = np.load(bio_file, allow_pickle=True).item()
    return bio_data['preprocessed_eeg_data'], bio_data['ch_names'], bio_data['times']


def load_synthetic_data(project_dir, sub, model_file):
    """Load synthetic EEG prediction data."""
    synthetic_file = os.path.join(project_dir, 'linear_results', f'sub-{sub:02d}', 
                                 'synthetic_eeg_data', f'{model_file}.npy')
    
    if not os.path.exists(synthetic_file):
        raise FileNotFoundError(f"Synthetic EEG file not found: {synthetic_file}")
    
    synthetic_data = np.load(synthetic_file, allow_pickle=True).item()
    return synthetic_data['synthetic_data']


def main():
    """Main function to run partial correlation analysis."""
    
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Compute partial correlations for EEG encoding models')
    parser.add_argument('--sub', type=int, default=4, help='Subject identifier')
    parser.add_argument('--project_dir', type=str, default=os.environ.get('EEG_FUSION_DATA', 'data'), 
                       help='Root project directory')
    parser.add_argument('--fusion_file', type=str, required=True,
                       help='Fusion model file name (without .npy extension)')
    parser.add_argument('--control_file', type=str, required=True,
                       help='Control model file name (without .npy extension)')
    parser.add_argument('--control_name', type=str, required=True,
                       help='Name of control modality (e.g., "text", "image")')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of bootstrap iterations')
    
    args = parser.parse_args()

    print(f"Partial Correlation Analysis")
    print(f"Subject: {args.sub}")
    print(f"Fusion model: {args.fusion_file}")
    print(f"Control model: {args.control_file}")
    print(f"Control modality: {args.control_name}")
    print("-" * 50)

    try:
        # Load biological EEG data
        print("Loading biological EEG data...")
        bio_test, ch_names, times = load_eeg_data(args.project_dir, args.sub)

        # Load fusion predictions
        print("Loading fusion predictions...")
        fusion_test = load_synthetic_data(args.project_dir, args.sub, args.fusion_file)

        # Load control predictions
        print("Loading control predictions...")
        control_test = load_synthetic_data(args.project_dir, args.sub, args.control_file)

        # Check if partial correlation results already exist
        results_dir = os.path.join(args.project_dir, 'linear_results', f'sub-{args.sub:02d}', 'partial_correlation')
        results_file = os.path.join(results_dir, f'partial_correlation_fusion_controlling_{args.control_name}.npy')
        
        if os.path.exists(results_file):
            print(f"Results already exist: {results_file}")
            print("Skipping computation. Use --force to overwrite.")
            return

        # Compute partial correlations
        print("Computing partial correlations...")
        partial_corr = plot_partial_correlation(
            fusion_test, control_test, bio_test, ch_names, times,
            args.control_name, args.sub, 
            os.path.join(args.project_dir, 'linear_results'),
            args.iterations
        )

        print(f"✅ Partial correlation analysis completed successfully!")
        print(f"Mean partial correlation: {np.nanmean(partial_corr):.3f}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()