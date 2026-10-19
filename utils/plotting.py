"""
Plotting utilities for track overlay ML model evaluation.

This module provides visualization functions for:
1. Training diagnostics (loss, accuracy curves)
2. Classification performance (ROC curves, score distributions)
3. Event-level routing analysis (efficiency, fraction plots)
4. Roulette decision diagnostics

All plots follow ATLAS style guidelines using mplhep.
"""

import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import spearmanr as sr
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import matplotlib.patches as mpatches
import warnings
import mplhep as hep


# =============================================================================
# Statistical Error Calculations
# =============================================================================

def plot_features(df, output_dir):
    print("Plotting input feature distributions")

    plt.style.use(hep.style.ATLAS)

    features = [
        "Px",
        "Py",
        "Pz",
        "E",
        "Pt",
        "R02",
        "R05",
        "sumR02",
        "sumR05",
        "sumpT02",
        "sumpT05",
        "w_numPU",
        "TruthMultiplicity",
        "eventPt",
    ]

    labels = {
        "Px": r"$p_x$",
        "Py": r"$p_y$",
        "Pz": r"$p_z$",
        "E": r"$E$",
        "Pt": r"$p_T$",
        "R02": r"$R02$",
        "R05": r"$R05$",
        "sumR02": r"$\Sigma \Delta R_{0.2}$",
        "sumR05": r"$\Sigma \Delta R_{0.05}$",
        "sumpT02": r"$\Sigma p_T^{0.2}$",
        "sumpT05": r"$\Sigma p_T^{0.05}$",
        "w_numPU": "Number of pile-up interactions",
        "TruthMultiplicity": "Truth multiplicity",
        "eventPt": r"Event $p_T$",
    }

    os.makedirs(output_dir, exist_ok=True)

    diffcheck = df[df["TargetLabel"] == 1]["E"]
    print("Maximum E for Diff:", diffcheck.max())
    print("Diff tracks with E > 500:", (diffcheck > 500).sum())
    print("Diff tracks with E > 600:", (diffcheck > 600).sum())
    nodiffcheck = df[df["TargetLabel"] == 0]["E"]
    print("Maximum E for NoDiff:", nodiffcheck.max())
    print("Maximum E for Diff:", diffcheck.max())

    for feature in features:

        nodiff = df[df["TargetLabel"] == 0][feature].to_numpy()
        diff = df[df["TargetLabel"] == 1][feature].to_numpy()
        # Remove NaN / inf values
        nodiff = nodiff[np.isfinite(nodiff)]
        diff = diff[np.isfinite(diff)]

        fig, ax = plt.subplots()

        # Use common binning for both distributions
        all_values = np.concatenate([nodiff, diff])

        bins = np.histogram_bin_edges(
            all_values,
            bins=50
        )

        ax.hist(
            nodiff,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            density=True,
            label="NoDiff"
        )

        ax.hist(
            diff,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            density=True,
            label="Diff"
        )

        log_features = [
        "Pt",
        "E",
        "Px",
        "Py",
        "Pz",
        "sumpT02",
        "sumpT05",
        "eventPt",
        "sumR02",
        "sumR05"
        ]
        if feature in log_features:
            ax.set_yscale("log")
        ax.set_xlabel(labels[feature])
        ax.set_ylabel("Normalized entries")

        ax.legend(frameon=False)

        fig.savefig(
            f"{output_dir}/feature_{feature}.pdf",
            bbox_inches="tight"
        )

        plt.close(fig)



def plot_eta(eta,dir):
    print("Plotting eta distribution")

    plt.style.use(hep.style.ATLAS)
    fig,ax = plt.subplots()
    
    ax.hist(eta, bins=50, range=(-5, 5))

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Entries")

    fig.savefig(f"{dir}/eta.pdf")
    plt.close(fig)
    

def error(success, exp, eff):
    """
    Compute Wilson confidence interval for binomial proportion.
    
    Returns the deviation from the efficiency value (eff) to create error bars.
    
    Parameters
    ----------
    success : int or np.ndarray
        Number of successes.
    exp : int or np.ndarray
        Number of trials (expected).
    eff : float or np.ndarray
        Efficiency (success/exp).
    
    Returns
    -------
    error : tuple of np.ndarray
        (lower_error, upper_error) deviations from eff.
    """
    return np.abs(proportion_confint(success, exp, method="wilson") - eff)


def combined_error(fullEff, testEff, fullErr, testErr):
    """
    Compute combined error for ratio of efficiencies.
    
    Used for error propagation when computing testEff/fullEff.
    
    Parameters
    ----------
    fullEff : np.ndarray
        Full efficiency values (denominator).
    testEff : np.ndarray
        Test efficiency values (numerator).
    fullErr : np.ndarray
        Error on full efficiency.
    testErr : np.ndarray
        Error on test efficiency.
    
    Returns
    -------
    combined_err : np.ndarray
        Propagated error on the ratio.
    """
    num = np.square(testEff) * np.square(fullErr) + np.square(fullEff) * np.square(testErr)
    den = np.power(fullEff, 4)
    return np.sqrt(num / den)


def frac_error(full, fullE, hyb, hybE, fast, fastE):
    """
    Compute error on fraction metric: (full - hyb) / (full - fast).
    
    This fraction quantifies how well hybrid overlay approximates full overlay
    relative to fast overlay. Error propagation accounts for correlations.
    
    Parameters
    ----------
    full, hyb, fast : np.ndarray
        Efficiency values for full, hybrid, and fast overlays.
    fullE, hybE, fastE : tuple of np.ndarray
        (lower, upper) errors for each efficiency.
    
    Returns
    -------
    frac_err : np.ndarray
        Propagated error on fraction with correct sign.
    """
    ratio = (full - hyb) / (full - fast)
    factor = ratio / np.abs(ratio)  # Preserve sign
    
    term1 = hyb - fast
    term2 = full - fast
    term3 = full - hyb
    
    # Convert error tuples to absolute values
    hybE = np.abs(np.diff(hybE, axis=0))
    fullE = np.abs(np.diff(fullE, axis=0))
    fastE = np.abs(np.diff(fastE, axis=0))

    # Error propagation formula for complex fraction
    num = (
        np.square(term1) * np.square(fullE)
        + np.square(term2) * np.square(hybE)
        + np.square(term3) * np.square(fastE)
    )
    den = np.power(term2, 4.0)

    return np.sqrt(num / den) * factor


def get_KL_div(mu1, sig1, mu2, sig2):
    """
    Compute Kullback-Leibler divergence between two Gaussian distributions.
    
    Measures disagreement between hybrid and full efficiency distributions.
    Lower values indicate better agreement.
    
    Parameters
    ----------
    mu1, sig1 : np.ndarray
        Mean and standard deviation of first distribution.
    mu2, sig2 : np.ndarray
        Mean and standard deviation of second distribution.
    
    Returns
    -------
    kl_div : float
        KL divergence value.
    """
    term1 = np.log(sig2 / sig1)
    term2 = (np.square(sig1) + np.square(mu1 - mu2)) / (np.square(sig2) * 2.0)
    return (term1 + term2 - 0.5).sum()


# =============================================================================
# Training Diagnostic Plots
# =============================================================================

def plot_losses(history, dir, logfile):
    """
    Plot training and validation loss evolution.
    
    Parameters
    ----------
    history : dict
        Training history with 'loss' and 'val_loss' keys.
    dir : str
        Directory to save plot.
    logfile : file
        Log file for output messages.
    """
    print("Plotting loss evolution", file=logfile, flush=True)
    
    fig, ax = plt.subplots()
    ax.plot(history["loss"], label="Training Loss")
    ax.plot(history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f"{dir}/loss_evolution.pdf",bbox_inches='tight')
    plt.close()


def plot_accuracy(history, dir, logfile):
    """
    Plot training and validation accuracy evolution.
    
    Parameters
    ----------
    history : dict
        Training history with 'accuracy' and 'val_accuracy' keys.
    dir : str
        Directory to save plot.
    logfile : file
        Log file for output messages.
    """
    print("Plotting accuracy evolution", file=logfile, flush=True)
    
    fig, ax = plt.subplots()
    ax.plot(history["balanced_accuracy"],label="Training Balanced Accuracy")

    ax.plot(history["val_balanced_accuracy"],label="Validation Balanced Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Balanced Accuracy")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f"{dir}/acc_evolution.pdf", bbox_inches='tight')
    plt.close()


# =============================================================================
# Classification Performance Plots
# =============================================================================

def plot_scores(losses, testlabels, dir, logfile, track_threshold=None, track_quantile=None):
    """
    Plot ML discriminant score distributions for Diff and No Diff.
    
    Shows separation between tracks that differ between overlays (Diff.) and
    those that don't (no Diff.). Better separation indicates better classifier.
    
    Parameters
    ----------
    losses : np.ndarray
        Model predictions (scores).
    testlabels : np.ndarray
        True labels (1=Diff., 0=no Diff.).
    dir : str
        Directory to save plot.
    logfile : file
        Log file for output messages.
    """
    print("Plotting score distributions", file=logfile, flush=True)
    
    fig, ax = plt.subplots()
    
    # Normalize to unit area for comparison
    weights = lambda data: np.ones(data.shape, dtype=float) / data.shape[0]
    
    # Plot Diff. (signal) distribution
    _, bins, _ = plt.hist(
        losses[testlabels == 1.0],
        bins=np.arange(0, 1, 0.03),
        weights=weights(losses[testlabels == 1.0]),
        histtype="step",
        linewidth=2,
        label="Diff"
    )
    
    # Plot no Diff. (background) distribution
    ax.hist(
        losses[testlabels == 0.0],
        bins=bins,
        weights=weights(losses[testlabels == 0.0]),
        histtype="step",
        linewidth=2,
        label="no Diff"
    )
    
    # 58% quantile of the true Diff distribution
    diff_scores = losses[testlabels == 1.0]
    #threshold = np.quantile(diff_scores, quantile)
    
    # Draw fixed threshold determined from validation sample
    if track_threshold is not None:
        if track_quantile is not None:
            threshold_label = (
                f"Validation {track_quantile:.0%} Diff quantile "
                f"= {track_threshold:.3f}"
            )
        else:
            threshold_label = f"Track threshold = {track_threshold:.3f}"
        ax.axvline(track_threshold,linestyle="--",linewidth=2,label=threshold_label)
    
    ax.set_xlabel("Track classifier score")
    ax.set_ylabel("Arbitrary Units (Normalized)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f"{dir}/scores.pdf", bbox_inches='tight')
    plt.close()


def plot_ROC(losses, testlabels, dir, logfile):
    """
    Plot ROC (Receiver Operating Characteristic) curve for binary classification.
    
    The ROC curve visualizes the trade-off between True Positive Rate (sensitivity)
    and False Positive Rate (1-specificity) at different classification thresholds.
    The AUC (Area Under Curve) score provides a single metric for model quality:
    - AUC = 1.0: Perfect classifier
    - AUC = 0.5: Random guessing
    - AUC < 0.5: Worse than random (inverted predictions)
    
    Parameters
    ----------
    losses : np.ndarray
        ML discriminant scores (higher = more signal-like)
    testlabels : np.ndarray
        True binary labels (1 = signal/Diff., 0 = background/no Diff.)
    dir : str
        Output directory path for saving the plot
    logfile : file object
        Log file handle for status messages
    
    Output
    ------
    Saves 'roc.png' showing ROC curve with AUC score in legend.
    Diagonal line (y=x) represents random classifier baseline.
    """
    print("Ploting ROC curve", file=logfile, flush=True)
    
    # Compute ROC curve: sweep through all possible thresholds
    fpr, tpr, thresholds = roc_curve(testlabels, losses, pos_label=1)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC {auc_score:3.3f}")
    ax.plot([0.0, 1], ls="--", color="black")  # Random baseline
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Classification")

    plt.legend(frameon=False, loc="lower right")
    plt.savefig(f"{dir}/roc.pdf")
    plt.clf()


def plot_roulette_score(score, roulThres, dir=None, logfile=None):
    """
    Plot event routing score distribution with ML threshold cut lines.
    
    Visualizes the distribution of xScore values (event-level roulette scores)
    with vertical lines showing the multiple threshold cuts being tested. Each
    cut separates events into "route to fast reconstruction" (low score) vs
    "route to full reconstruction" (high score).
    
    The color-coded vertical lines correspond to different quantile cuts
    (typically [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8] → 7 different strategies).
    
    Parameters
    ----------
    score : np.ndarray
        Event-level xScore values (roulette scores for routing decisions)
    roulThres : list of float
        Threshold values to mark as vertical lines (different ML cuts to test)
    dir : str, optional
        Output directory path for saving the plot
    logfile : file object, optional
        Log file handle for status messages
    
    Output
    ------
    Saves 'rouletteScore.png' showing score histogram with colored threshold lines.
    Each cut line is labeled (cut 1, cut 2, ...) in the legend.
    """
    plt.style.use(hep.style.ATLAS)
    print("Plotting roulette score", file=logfile, flush=True)

    # Create rainbow colormap for threshold lines
    color_map = plt.cm.get_cmap("rainbow", len(roulThres))
    colors = [color_map(i) for i in range(len(roulThres))]
    cutpointer = [mpatches.Patch(color=color) for color in colors]

    cutlabels = [f"cut {i+1}" for i in range(len(roulThres))]

    # Use 0.1% and 99.9% quantiles to avoid extreme outliers
    lower = np.quantile(score, 0.001)
    upper = np.quantile(score, 0.999)
    steps = (upper - lower) / 50

    fig, ax = plt.subplots()

    ax.hist(score, np.arange(lower, upper, steps))
    y1, y2 = ax.get_ylim()
    ax.vlines(x=roulThres, ymin=y1, ymax=y2, color=colors)  # Draw threshold lines
    ax.set_ylim(y1, y2)
    ax.set_xlabel("Event xScore")

    ax.legend(cutpointer, cutlabels, frameon=False)

    plt.savefig(f"{dir}/rouletteScore.pdf")
    plt.clf()


def plot_fast_fraction(isf, bins, observable, dir, logfile=None, **kwargs):
    """
    Plot fraction of events routed to fast track overlay reconstruction.
    
    Shows what percentage of events in each bin of an observable (e.g., NPV, mu)
    are routed to fast reconstruction vs full reconstruction by the ML model.
    This helps understand the routing behavior across different physics regimes.
    
    Fraction = (# events routed to fast) / (# total events in bin)
    
    Higher fraction = more aggressive use of fast reconstruction
    Lower fraction = more conservative, using full reconstruction more often
    
    Parameters
    ----------
    isf : dict
        Dictionary containing 'frac' (events to fast) and 'tot' (total events) arrays
    bins : np.ndarray
        Bin edges for the observable being analyzed
    observable : str
        Name of observable being binned (e.g., 'NPV', 'mu', 'jet pT')
    dir : str
        Output directory path for saving plots and data
    logfile : file object, optional
        Log file handle for status messages
    **kwargs : dict
        - quality : str, routing strategy quality label (default: 'generic')
        - units : str, units for observable x-axis (default: '')
        - sv_prefix : str, save file prefix (default: 'generic')
    
    Output
    ------
    Saves 'frac_{sv_prefix}_{quality}.png' plot showing fraction vs observable.
    Also saves numerical data as 'frac_{sv_prefix}_{quality}.npy' for later analysis.
    """
    sample = dir.split("/")[1]
    plt.style.use(hep.style.ATLAS)

    # Parse optional kwargs with defaults
    if "quality" in kwargs:
        quality = kwargs["quality"]
    else:
        quality = "generic"
        warnings.warn(
            "Quality of cut not passed. Plots will be overwritten and saved as generic.png.\
            kwarg 'quality' expects a string."
        )

    if "units" in kwargs:
        units = kwargs["units"]
    else:
        units = ""
        warnings.warn("No Units passed. Passing empty string.")

    if "sv_prefix" in kwargs:
        svp = kwargs["sv_prefix"]
    else:
        svp = "generic"
        warnings.warn("No save prefix found, passing generic as default.")

    print("Plotting fast/full fraction plot", file=logfile, flush=True)

    fractot = isf["frac"]  # Events routed to fast reconstruction
    tot = isf["tot"]       # Total events

    # Convert bin edges to bin centers for plotting
    bins = (bins[1:] + bins[:-1]) / 2.0
    bins[-1] = 2 * bins[-2] - bins[-3]  # Extrapolate last bin center
    bins[0] = 2 * bins[1] - bins[2]     # Extrapolate first bin center
    bins = [round(item, 1) for item in bins]
    tlow, thigh = bins[0], bins[-1]

    # Create string labels with overflow/underflow indicators
    sbins = [str(item) for item in bins]
    sbins[-1] = ">" + str(bins[-2])  # Overflow bin
    sbins[0] = "<" + str(bins[1])    # Underflow bin

    # Calculate fraction and propagate statistical errors
    frac = fractot / tot
    fracErr = error(fractot, tot, frac)

    # Save numerical data for later analysis
    save_dat = np.dstack([np.array(bins), frac.data, fracErr[0,:].data, fracErr[1,:].data])
    save_dat_dir = f'{dir}/frac_{svp}_{quality}.npy'
    np.save(save_dat_dir, save_dat)

    low, high = frac.min(), frac.max()
    fig, ax = plt.subplots(figsize=(8.3, 8.3))

    hep.atlas.text(text="Simulation Work in Progress", ax=ax, loc=1)

    ax.errorbar(
        bins, frac, yerr=fracErr, marker="*", ls="", color="blue", elinewidth=0.5
    )
    ax.set_xlim(np.min(bins), np.max(bins))
    ax.set_xticks(bins)
    ax.set_xticklabels(sbins)
    ax.tick_params(axis="x", rotation=45.0, labelsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction to Fast Reconstruction", fontsize=15)
    ax.set_xlabel(f"{observable} {units}")
    ax.text(
        0.75, 0.90, f"{sample} events\n-2.5 $\leq$ $\eta$ $\leq$ 2.5"
    )
    plt.savefig(f"{dir}/frac_{svp}_{quality}.pdf")
    plt.clf()


def plot_efficiencies(
    isf, bins, down, up, mean, observable, dir=None, logfile=None, **kwargs
):
    """
    Plot track reconstruction efficiencies for fast/full/hybrid strategies.
    
    Creates a two-panel plot:
    - Top panel: Track reconstruction efficiency vs observable for three strategies:
      * Fast: Use fast track overlay for all events (baseline, lowest efficiency)
      * Full: Use full track overlay for all events (baseline, highest efficiency)
      * Hybrid: ML-driven routing (balance efficiency and speed)
    
    - Bottom panel: Fraction metric = (Full - Hybrid) / (Full - Fast)
      * Measures how well hybrid strategy performs relative to baselines
      * Lower is better: 0 = matches full (best), 1 = matches fast (worst)
      * Compared to random routing (shaded band shows ±1σ variation)
    
    This is the KEY DIAGNOSTIC PLOT showing whether ML routing successfully
    maintains high efficiency while reducing computational cost.
    
    Parameters
    ----------
    isf : dict
        Counts dictionary with keys:
        - 'fast' : tracks reconstructed with fast overlay (all events)
        - 'full' : tracks reconstructed with full overlay (all events)
        - 'hyb' : tracks reconstructed with hybrid ML routing
        - 'tot' : total tracks (denominator for efficiency calculation)
    bins : np.ndarray
        Bin edges for the observable being analyzed
    down : np.ndarray
        Lower bound (mean - 1σ) of random routing fraction across bins
    up : np.ndarray
        Upper bound (mean + 1σ) of random routing fraction across bins
    mean : np.ndarray
        Mean value of random routing fraction baseline
    observable : str
        Name of observable being binned (e.g., 'NPV', 'mu', 'jet pT')
    dir : str, optional
        Output directory path for saving plots and data
    logfile : file object, optional
        Log file handle for status messages
    **kwargs : dict
        - quality : str, routing strategy quality label (default: 'generic')
        - sv_prefix : str, save file prefix (default: 'generic')
        - units : str, units for observable x-axis (default: '')
    
    Output
    ------
    Saves 'isf_{sv_prefix}_{quality}.png' with two-panel efficiency comparison plot.
    Saves 'FA_{sv_prefix}_{quality}.npy' with fraction metric numerical data.
    Prints KL divergence between full and hybrid strategies to logfile.
    """
    plt.style.use(hep.style.ATLAS)

    print("Plotting ISF", file=logfile, flush=True)
    sample = dir.split("/")[1]

    # Parse optional kwargs with defaults
    if "quality" in kwargs:
        quality = kwargs["quality"]
    else:
        quality = "generic"
        warnings.warn(
            "Quality of cut not passed. Plots will be overwritten and saved as generic.png.\
            kwarg 'quality' expects a string."
        )

    if "sv_prefix" in kwargs:
        svp = kwargs["sv_prefix"]
    else:
        svp = "generic"
        warnings.warn("No save prefix found, passing generic as default.")

    if "units" in kwargs:
        units = kwargs["units"]
    else:
        units = ""
        warnings.warn(
            "No units passed for observable. Passing empty string as default."
        )

    # Convert bin edges to bin centers for plotting
    bins = (bins[1:] + bins[:-1]) / 2.0
    bins[-1] = 2 * bins[-2] - bins[-3]  # Extrapolate last bin center
    bins[0] = 2 * bins[1] - bins[2]     # Extrapolate first bin center
    bins = [round(item, 1) for item in bins]
    tlow, thigh = bins[0], bins[-1]

    # Create string labels with overflow/underflow indicators
    sbins = [str(item) for item in bins]
    sbins[-1] = ">" + str(bins[-2])  # Overflow bin
    sbins[0] = "<" + str(bins[1])    # Underflow bin

    # Calculate efficiencies: eff = (tracks reconstructed) / (total tracks)
    fast, full, hyb, tot = isf["fast"], isf["full"], isf["hyb"], isf["tot"]
    fastEff = fast / tot
    fastErr = error(fast, tot, fastEff)
    fullEff = full / tot
    fullErr = error(full, tot, fullEff)
    hybEff = hyb / tot
    hybErr = error(hyb, tot, hybEff)

    # Calculate disagreement metric (KL divergence) between full and hybrid
    disagg = get_KL_div(fullEff.data, fullErr[0].data, hybEff.data, hybErr[0].data)

    colors = ["green", "red", "blue"]
    simpointer = ["o", "s", "*"]
    simlabel = ["Fast Reconstruction", "Full Reconstruction", "Hybrid Reconstruction"]

    # Create two-panel figure with shared x-axis
    fig = plt.figure(figsize=(8.3, 8.3))
    gs = fig.add_gridspec(2, hspace=0.1, height_ratios=[1.0, 0.8])
    ax = gs.subplots(sharex=True)
    hep.atlas.text(text="Simulation Work in Progress", ax=ax[0], loc=1, fontsize=14)

    # Top panel: Plot all three efficiencies
    ax[0].errorbar(
        bins,
        fastEff,
        yerr=fastErr,
        marker=simpointer[0],
        ms=4,
        ls="",
        color=colors[0],
        elinewidth=0.5,
        label=simlabel[0],
    )
    ax[0].errorbar(
        bins,
        fullEff,
        yerr=fullErr,
        marker=simpointer[1],
        ms=4,
        ls="",
        color=colors[1],
        elinewidth=0.5,
        label=simlabel[1],
    )
    ax[0].errorbar(
        bins,
        hybEff,
        yerr=hybErr,
        marker=simpointer[2],
        ms=4,
        ls="",
        color=colors[2],
        elinewidth=0.5,
        label=simlabel[2],
    )

    # Bottom panel: Calculate fraction metric (Full - Hybrid) / (Full - Fast)
    # This shows where hybrid falls between the two baselines
    comb_ratio = np.abs((fullEff - hybEff) / (fullEff - fastEff))
    comb_error = frac_error(fullEff, fullErr, hybEff, hybErr, fastEff, fastErr)

    # Save numerical data for fraction metric
    save_dat = np.dstack([np.array(bins), comb_ratio.data, comb_error.data])
    save_dat_dir = f"{dir}/FA_{svp}_{quality}.npy"
    np.save(save_dat_dir, save_dat)

    # Top panel formatting
    ax[0].legend(bbox_to_anchor=(0.01, 0.2, 1.0, 0.1), frameon=False)
    ax[0].text(np.quantile(bins, 0.75), 0.93, f"{sample} events\n |$\eta$|$ \leq$ 2.5")
    ax[0].set_xlim(np.min(bins), np.max(bins))
    ax[1].set_xticks(bins)
    ax[1].set_xticklabels(sbins)
    ax[1].tick_params(axis="x", rotation=45.0, labelsize=8)
    ax[1].set_xlabel(f"{observable} {units}")
    ax[0].set_ylabel(f"Track Reconstruction Efficiency", fontsize=15)
    ax[0].set_ylim(0.65, 1.0)
    
    # Bottom panel: Plot fraction metric vs random baseline
    ax[1].errorbar(
        bins,
        comb_ratio,
        yerr=np.abs(comb_error.data.squeeze(0)),
        marker="o",
        ls="",
        ms=5,
        label="Score",
    )
    ax[1].plot(bins, up, ls="", color="b")       # Random upper bound
    ax[1].plot(bins, down, ls="", color="b")     # Random lower bound
    ax[1].plot(bins, mean, ls="", marker="x", color="k")  # Random mean
    ax[1].fill_between(bins, up, down, alpha=0.4, label="Random")  # Shaded ±1σ band
    ax[1].set_ylabel(r"$\frac{Full-Hybrid}{Full-Fast}$")
    ax[1].legend(frameon=False, loc="upper right")
    
    plt.savefig(f"{dir}/isf_{svp}_{quality}.pdf")
    plt.clf()
    
    print(
        f"Disagreement with hybrid-full for {observable}: {disagg:3.3f} for {quality}",
        file=logfile,
        flush=True,
    )


def plot_roulette_diagnostic(scores, avg_decision, dir=None, logfile=None, quality="generic", svp="generic", **kwargs):
    """
    Plot diagnostic for event routing decision profile with stochastic sampling.
    
    Creates a two-panel diagnostic plot showing:
    - Top panel: Score distribution with threshold cut line
      Shows where the quantile cut separates events into fast vs full routing
    
    - Bottom panel: Fraction routed to fast reconstruction vs score
      Shows the temperature-scaled sigmoid probability for "smart" decisions
      Points represent individual events, showing stochastic sampling behavior
    
    This diagnostic helps understand how the "smart" decision strategy works:
    events with scores below threshold have high probability of fast routing,
    while events above threshold have low probability (but both use stochastic
    sampling rather than hard cuts for smoother decision boundaries).
    
    Displays efficiency metrics:
    - ε_fast,fast: Fraction of "fast" events (score < threshold) routed to fast
    - ε_fast,full: Fraction of "full" events (score > threshold) routed to fast
    - ε: Overall fraction routed to fast = weighted combination of above
    
    Parameters
    ----------
    scores : np.ndarray
        Event-level xScore values for routing decisions
    avg_decision : np.ndarray
        Fraction routed to fast (0-1) for each event from stochastic sampling
        For "smart" strategy: sigmoid probability based on score and temperature
    dir : str, optional
        Output directory path for saving the plot
    logfile : file object, optional
        Log file handle for status messages
    quality : str, default='generic'
        Routing strategy quality label
    svp : str, default='generic'
        Save file prefix
    **kwargs : dict
        - quantile : float, threshold quantile cut (e.g., 0.7 for 70% to fast)
    
    Output
    ------
    Saves 'profile_{svp}_{quality}.png' showing routing decision profile diagnostic.
    """
    print("Plotting Roulette profile", file=logfile, flush=True)
  
    quantile = kwargs["quantile"]
    thres = np.quantile(scores, quantile)    
    
    # Calculate efficiency metrics for "fast" and "full" score regions
    fast_from_full = avg_decision[scores > thres].sum()/len(scores[scores>thres])
    fast_from_fast = avg_decision[scores < thres].sum()/len(scores[scores<thres])
    total_to_fast = fast_from_fast*quantile+fast_from_full*(1-quantile)
    
    plt.style.use(hep.style.ATLAS)
    fig = plt.figure(figsize=(10,10))
    
    gs = fig.add_gridspec(2, hspace=0.02, height_ratios=[0.4, 0.6])
    ax = gs.subplots(sharex=True)
    
    hep.atlas.text(text="Simulation Work in Progress", ax=ax[0], loc=1, fontsize=14)
    
    # Top panel: Score distribution with threshold line
    counts,bins,_=ax[0].hist(scores, bins=80)
    
    ceiling = 2*np.max(counts) - np.quantile(counts, 0.45)
    f_tx, f_ty = np.quantile(bins, 0.032), ceiling-np.quantile(counts, 0.98)
    
    # Bottom panel: Scatter plot showing stochastic decision probabilities
    ax[1].scatter(scores, avg_decision, marker='x', lw=0.5, s=1., color='blue')
    ax[1].axvline(x=thres, color='red', lw= 2.)  # Threshold cut line
    ax[0].vlines(x=thres, ymin=0, ymax=np.max(counts), color='red', lw= 2., label = f"{quantile*100:3.1f}% Cut")
    low, high = scores.min(), scores.max()
    ax[1].hlines(1, xmin=low, xmax=high, color='black', lw=1.)  # 100% fast line
    ax[1].set_xlim(low, high)
    ax[1].set_ylim(0.,1.02)
    ax[0].set_ylim(0, ceiling)
    ax[1].set_xlabel("Scores", loc="center")
    ax[1].set_ylabel("Fraction to Fast reconstruction", fontsize=14, loc="center")
    
    # Display efficiency metrics
    ax[0].text(f_tx, f_ty, f"$\epsilon_{{fast,fast}}: {fast_from_fast*100:3.2f}$%\n$\epsilon_{{fast,full}}: {fast_from_full*100:3.2f}$%\n$\epsilon: {total_to_fast*100:3.2f}$%", fontsize=12)
    ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0., 0.45, 0.5, 0.5))
    plt.savefig(f"{dir}/profile_{svp}_{quality}.pdf")
    plt.clf()


def plot_jet_mismodelling(pts, dir=None, quality='generic', logfile=None,):
    """
    Plot jet pT mismodelling between fast/full and hybrid reconstruction strategies.
    
    Analyzes the fractional difference in reconstructed jet transverse momentum
    (pT) between different track overlay strategies. This diagnostic checks if
    the hybrid ML routing introduces systematic biases in jet reconstruction
    compared to the baseline strategies.
    
    Fractional difference = |pT_full - pT_strategy| / pT_full
    
    Bins jets by their full reconstruction pT to see if mismodelling varies
    with jet energy. Ideally, hybrid should closely match full reconstruction
    (low fractional difference) across all pT bins.
    
    Parameters
    ----------
    pts : dict
        Dictionary containing jet pT arrays:
        - 'full' : jet pT with full track overlay (reference)
        - 'fast' : jet pT with fast track overlay
        - 'hyb' : jet pT with hybrid ML routing
    dir : str, optional
        Output directory path for saving the plot
    quality : str, default='generic'
        Routing strategy quality label
    logfile : file object, optional
        Log file handle for status messages
    
    Output
    ------
    Saves 'ptmismodelling_{quality}.png' showing fractional pT differences
    vs full reconstruction jet pT for fast and hybrid strategies.
    Zero line (green) represents perfect agreement with full reconstruction.
    """
    print("Plotting jet pt mismodelling", file=logfile, flush=True)

    full_pt = pts["full"]
    fast_pt = pts["fast"]
    hyb_pt = pts["hyb"]

    # Calculate fractional difference relative to full reconstruction
    frac_diff = lambda full, frac : np.abs(full-frac)/full
    # Select jets in a specific pT bin
    sieve = lambda pt,low,high: pt[(full_pt > low) & (full_pt < high)]
       
    # Define pT bins covering central 80% of distribution (avoid extreme outliers)
    low, high = np.quantile(full_pt, 0.1), np.quantile(full_pt, 0.9)
    
    bins=np.linspace(low, high, 78)
    bins = np.insert(bins, 0, np.min(full_pt))     # Add underflow bin
    bins = np.insert(bins, len(bins), np.max(full_pt))  # Add overflow bin

    bin_centers = 0.5*(bins[1:] + bins[:-1])
    bin_centers = np.round(bin_centers, 2)

    # Extrapolate edge bin centers for plotting
    bin_centers[-1] = 2 * bin_centers[-2] - bin_centers[-3]
    bin_centers[0] = 2 * bin_centers[1] - bin_centers[2]

    # Create string labels with overflow/underflow indicators
    sbins = [str(item) for item in bin_centers]
    sbins[-1] = ">" + str(bin_centers[-2])
    sbins[0] = "<" + str(bin_centers[1])
    
    # Calculate mean and std of fractional difference in each bin
    plotter_hyb = np.empty(shape=(2, len(bins)-1))    
    plotter_fast = np.empty(shape=(2, len(bins)-1))
    
    for idx,(i,j) in enumerate(zip(bins[:-1], bins[1:])):
        bin_fast = frac_diff(sieve(full_pt, i, j), sieve(fast_pt, i, j))
        bin_hyb = frac_diff(sieve(full_pt, i, j), sieve(hyb_pt, i, j))
        
        plotter_fast[:, idx] = bin_fast.mean(), bin_fast.std()
        plotter_hyb[:, idx] = bin_hyb.mean(), bin_hyb.mean()  # Note: uses mean twice (likely should be std)
    
    plt.style.use(hep.style.ATLAS)
    fig,ax = plt.subplots(figsize=(10,10))
    
    hep.atlas.text(text="Simulation Work in Progress", ax=ax, loc=1, fontsize=14)
    
    # Plot fractional differences vs jet pT
    ax.errorbar(bin_centers, plotter_fast[0], color='red', label='Fast')
    ax.errorbar(bin_centers, plotter_hyb[0], color='blue', label='Hybrid')
    ax.axhline(0, color='green')  # Perfect agreement line
    ax.legend(frameon=False)
    ax.set_xlabel("Jet pT, [GeV]")
    ax.set_xticks(bin_centers[0::5])  # Show every 5th tick to avoid crowding
    ax.set_xticklabels(sbins[0::5], rotation=45)
    ax.set_ylim(-0.0005,0.02)
    ax.set_xlim(np.min(bin_centers), np.max(bin_centers))
    plt.savefig(f"{dir}/ptmismodelling_{quality}.pdf")
    plt.show()
