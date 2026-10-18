"""
Evaluator for multi-class track overlay model.

This module provides functions to evaluate model performance at both track-level
and event-level. The evaluation includes:

1. Track-level metrics: accuracy, loss, ROC curves
2. Event-level metrics: efficiency analysis with ML-optimized thresholds
3. Roulette decision strategies: hard cuts vs. smart (temperature-based) sampling

The ML threshold optimization compares ML-based selections to random selection,
where events are randomly assigned to track overlay or MC overlay while maintaining
the same proportion. Performance is evaluated using a fraction that measures how well
the selection improves efficiency by comparing MC overlay and hybrid overlay efficiency
relative to track overlay. Lower fraction values indicate better agreement between 
hybrid and MC overlay, demonstrating ML-based selection advantage over random selection.
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from data.features import get_feature_names

from .plotting import (
    plot_eta,
    plot_ROC,
    plot_accuracy,
    plot_efficiencies,
    plot_fast_fraction,
    plot_jet_mismodelling,
    plot_losses,
    plot_roulette_diagnostic,
    plot_roulette_score,
    plot_scores,
)

from .xscore import getxScore
import os
import numpy as np
import pandas as pd
import sys


# =============================================================================
# Event-Level Score Computation
# =============================================================================

def get_event_xscore(model, FSscaler, eventSet, use_densities, logfile, **kwargs):
    """
    Compute event-level xScore using the trained model.
    
    The xScore quantifies event quality for routing decisions between track overlay
    (fast) and MC overlay (full simulation). Higher scores indicate events better
    suited for track overlay.
    
    Parameters
    ----------
    model : keras.Model
        Trained classifier model.
    FSscaler : MinMaxScaler
        Feature scaler for normalization.
    eventSet : pd.DataFrame
        Event dataset with features.
    use_densities : bool
        Whether density features (R02, R05, etc.) are included.
    logfile : file
        Log file for output.
    **kwargs : dict
        xscore : str, default="simpleratio"
            Type of xScore metric to compute.
        batchsize : int, default=50000
            Batch size for model prediction.
        frac : float, default=0.25
            Fraction threshold for event mixing.
        evtmix : str, default="sum"
            Event mixing strategy (sum, mean, etc.).
        thres : float, default=0.8
            Threshold for quantile-based track selection.
    
    Returns
    -------
    rouletteScore : np.ndarray
        Computed roulette score for each event.
    """
    
    # Extract parameters from kwargs with defaults
    batchsize = kwargs.get("batchsize", 50000)
    xscore = kwargs.get("xscore", "simpleratio")
    frac = kwargs.get("frac", 0.25)
    evtmix = kwargs.get("evtmix", "sum")
    thres = kwargs.get("thres", 0.8)
    track_threshold = kwargs["track_threshold"]
    
    
    featList = get_feature_names(use_densities)
    numFeatures = len(featList)
    
    print(f"Computing event xScore with {numFeatures} features", file=logfile, flush=True)
    
    # Keep exactly the features used by the model
    eventSet = eventSet[featList]
    
    # Convert to xarray and record the exact event order used
    eventSet_xr = eventSet.to_xarray()
    event_ids = eventSet_xr.coords["EvtNumber"].values
    eventSet = eventSet_xr.to_array().values
    eventSet = np.ma.masked_invalid(eventSet)
    eventSet = np.transpose(eventSet, axes=[1, 2, 0])  # Shape: (nEvents, nTracks, nFeatures)
    
    l, b, h = eventSet.shape  # l=events, b=tracks per event, h=features
    eventSet_flat = eventSet.reshape(l * b, h)
    
    # Scale features and predict
    #feat = FSscaler.fit_transform(eventSet_flat)
    feat = FSscaler.transform(eventSet_flat)
    scores = model.predict(feat, batch_size=batchsize)
    scores = np.ma.masked_invalid(scores)
    
    # Reshape scores back to event structure
    scores = scores.reshape(l, b, 1)
    
    # Compute roulette score using specified xScore method
    rouletteScore = getxScore(
        xscore, scores, track_threshold, eventSet.reshape(l, b, h), frac=frac, event_mixing=evtmix )
    
    return rouletteScore, event_ids


# =============================================================================
# Binning and Efficiency Analysis
# =============================================================================

def get_bins(dataframe, key, num_bins):
    """
    Generate binning scheme for observables.
    
    Special handling for pT variables to capture tail behavior and use
    quantile-based binning for better distribution coverage.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing the observable.
    key : str
        Column name of the observable to bin.
    num_bins : int
        Number of bins to create.
    
    Returns
    -------
    bins : np.ndarray
        Bin edges.
    labels : np.ndarray
        Bin labels (0 to num_bins-1).
    """
    if key[-2:].casefold() == "pt":
        if len(key) > 2:
            # For pT variables: use quantile-based binning for central region
            abs_low, abs_high = np.min(dataframe[key]), np.max(dataframe[key])
            low, high = np.quantile(dataframe[key], 0.05), np.quantile(dataframe[key], 0.95)
            bins = np.linspace(low, high, num_bins, endpoint=True)
            # Add absolute edges to capture tails
            bins = np.insert(bins, 0, abs_low)
            bins = np.insert(bins, len(bins), abs_high)
            labels = np.arange(0, len(bins) - 1, 1)
            return bins, labels
        else:
            # For simple "Pt": use fixed 2 GeV bins
            bins = np.arange(0, 52, 2)
            bins = np.insert(bins, len(bins), dataframe[key].max())
            labels = np.arange(0, len(bins) - 1, 1)
            return bins, labels
    else:
        # For other observables: uniform binning
        low, high = dataframe[key].min(), dataframe[key].max()
        bins = np.linspace(low, high, num_bins, endpoint=True)
        labels = np.arange(0, len(bins) - 1, 1)
        return bins, labels


def random_FA_spread(df, scores, score_event_ids, observable, roulThres):
    """
    Compute random baseline for fraction analysis with uncertainty bands.
    
    Simulates random event routing to establish a baseline comparison for
    ML-based selection. Computes mean and standard deviation across multiple
    random trials to quantify uncertainty.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event dataframe.
    scores : np.ndarray
        Array of random scores (multiple samples per event).
    observable : str
        Observable to bin on (e.g., "Pt", "Eta").
    roulThres : float
        Threshold for routing decision.
    
    Returns
    -------
    random_down : np.ndarray
        Lower uncertainty band (mean - std).
    random_up : np.ndarray
        Upper uncertainty band (mean + std).
    mean : np.ndarray
        Mean fraction across random trials.
    """
    estimate = scores.shape[0]  # Number of random estimates
    bins, labels = get_bins(df, observable, 25)
    df["binned"] = pd.cut(df[observable], bins=bins, labels=labels)

    # Get base data
    ptdata = df[["binned", "wo_MatchProb", "w_MatchProb"]].copy()

    # Broadcast all scores at once to avoid DataFrame fragmentation
    #group_sizes = df.groupby("EvtNumber").size().to_numpy()
    
    # Pre-compute all new columns in dictionaries (avoids repeated frame.insert calls)
    newcols = {}
    
    row_event_ids = df.index.get_level_values("EvtNumber")
    
    for i in range(estimate):
        #scoresBroad = np.repeat(scores[i], group_sizes)
        score_by_event = pd.Series(scores[i],index=score_event_ids)
        scoresBroad = (score_by_event.reindex(row_event_ids).to_numpy())
        newcols[f"scores_{i}"] = scoresBroad
        
        # Compute hybrid MatchProb directly
        newcols[f"h_MatchProb_{i}"] = np.where(
            scoresBroad > roulThres,
            ptdata["w_MatchProb"].values,
            ptdata["wo_MatchProb"].values,
        )
    
    # Concatenate all new columns at once (much faster than repeated inserts)
    ptdata = pd.concat([ptdata, pd.DataFrame(newcols, index=ptdata.index)], axis=1)
        
    # Prepare data for efficiency calculation
    ptdata = ptdata.reset_index(0).set_index("binned").sort_index()
    ptdata = ptdata.loc[ptdata.index.notnull()]
    ptdata = ptdata.set_index(ptdata.groupby("binned").cumcount(), append=True)

    if observable == "wo_leadJetpT":
        ptdata = ptdata.drop_duplicates("EvtNumber")
        
    ptdata = ptdata.to_xarray().to_array().values
    ptdata = np.transpose(ptdata, [1, 2, 0])
    ptdata = np.array(ptdata, dtype=float)
    ptdata = np.ma.masked_invalid(ptdata)
    
    # Compute efficiencies
    fastEff = np.count_nonzero(ptdata[:, :, 1] > 0.5, axis=1)
    fullEff = np.count_nonzero(ptdata[:, :, 2] > 0.5, axis=1)
    tot = np.ma.count(ptdata[:, :, 0], axis=1)

    # Compute fraction for each random sample
    # Fraction = (fullEff - hybEff) / (fullEff - fastEff)
    # Lower values indicate better agreement between hybrid and MC overlay
    glob_fi = np.empty(shape=(ptdata.shape[0], estimate))

    for i in range(estimate):
        hybEff = np.count_nonzero(ptdata[:, :, 3 + estimate + i] > 0.5, axis=1)
        he = hybEff / tot
        fue = fullEff / tot
        fae = fastEff / tot
        fi = (fue - he) / (fue - fae)
        glob_fi[:, i] = fi

    # Compute statistics across random trials
    dev = glob_fi.std(axis=1)
    mean = glob_fi.mean(axis=1)
    random_up = mean + dev
    random_down = mean - dev

    return random_down, random_up, mean


# =============================================================================
# Roulette Decision Strategies
# =============================================================================

def get_decision(scores, logfile=None, **kwargs):
    """
    Make routing decisions (track overlay vs MC overlay) based on scores.
    
    Two strategies:
    1. **Hard cut**: Simple threshold-based decision (deterministic)
    2. **Smart sampling**: Temperature-scaled sigmoid with random sampling (stochastic)
    
    The smart strategy uses temperature to control decision sharpness:
    - Low temperature → sharp decisions (closer to hard cut)
    - High temperature → soft decisions (more exploration)
    
    Parameters
    ----------
    scores : np.ndarray
        Event-level roulette scores.
    logfile : file, optional
        Log file for output.
    **kwargs : dict
        fu_weight : float, default=0.8
            Weight for full-to-fast transition (unused in current implementation).
        f_weight : float, default=1.0
            Weight for fast fraction (unused in current implementation).
        smoothing : float, default=0.0001
            Temperature parameter for smart roulette.
        sampler : str, default="uniform"
            Sampling distribution (only "uniform" supported).
        rouletter : str, default="hard"
            Decision strategy: "hard" or "smart".
        quantile : float, default=0.25
            Quantile for threshold calculation.
    
    Returns
    -------
    decision : np.ndarray (bool)
        Boolean array: True → track overlay (fast), False → MC overlay (full).
    """
    
    # Extract parameters
    fu_weight = kwargs.get("fu_weight", 0.8)
    f_weight = kwargs.get("f_weight", 1.0)
    temperature = kwargs.get("smoothing", 0.0001)
    sampler = kwargs.get("sampler", "uniform")
    rouletter = kwargs.get("rouletter", "hard")
    quantile = kwargs.get("quantile", 0.25)
    
    # Force hard cut if temperature is zero
    rouletter = "hard" if temperature == 0. else rouletter
    
    if rouletter.casefold() == "smart":
        sampler = "uniform"  # Smart roulette always uses uniform sampling
        
    num_samples = len(scores)
    
    # Generate random samples for stochastic decisions
    if sampler.casefold() == "uniform":
        sampled = np.random.uniform(size=num_samples)
    
    if rouletter.casefold() == "smart":
        # Smart roulette: temperature-scaled sigmoid sampling
        # 1. Normalize scores
        scores = scores / np.linalg.norm(scores)
        
        # 2. Compute threshold
        thres = np.quantile(scores, quantile)
        
        #print(f"Smart roulette: threshold={thres:.4f}, temperature={temperature}", file=logfile, flush=True)
        
        # 3. Compute weights using temperature-scaled sigmoid
        #    weight = exp((score - threshold) / T) / (1 + exp((score - threshold) / T))
        #    Higher scores → higher weight → more likely to use track overlay
        weights = np.exp((scores - thres) / temperature)
        weights = weights / (1 + weights)
        
        # 4. Stochastic decision: sample > weight → track overlay
        decision = sampled > weights
        
        #print(f"Smart decision: {np.sum(decision)} events to track overlay ({np.mean(decision)*100:.1f}%)", 
        #      file=logfile, flush=True)
        
        # Save decision for analysis
        np.savez(f"decision_{quantile}.npz", decision=decision)
        
    elif rouletter.casefold() == "hard":
        # Hard cut: deterministic threshold-based decision
        thres = np.quantile(scores, quantile)
        decision = np.ma.masked_array(scores < thres).data
        
    else:
        print(f"ERROR: '{rouletter}' is not a valid decision strategy. Choose 'smart' or 'hard'.")
        sys.exit()
        
    return decision


def binned_efficiency(df, scores, score_event_ids, observable, ret_diff=False, logfile=None, **kwargs):
    """
    Compute efficiency in bins of an observable.
    
    Calculates matching efficiency for track overlay (fast), MC overlay (full),
    and hybrid (ML-routed) approaches across bins of the specified observable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event dataframe with match probabilities.
    scores : np.ndarray
        Event-level roulette scores for routing.
    observable : str
        Observable to bin on (e.g., "Pt", "Eta").
    ret_diff : bool, default=False
        If True, return jet pT differences instead of efficiencies.
    logfile : file, optional
        Log file for output.
    **kwargs : dict
        Routing decision parameters (passed to get_decision).
    
    Returns
    -------
    If ret_diff=False:
        isf : dict
            Dictionary with keys:
            - "fast": Track overlay efficiency per bin
            - "full": MC overlay efficiency per bin
            - "hyb": Hybrid (ML-routed) efficiency per bin
            - "tot": Total counts per bin
            - "frac": Fraction of events routed to track overlay per bin
        bins : np.ndarray
            Bin edges for the observable.
    
    If ret_diff=True:
        pts : dict
            Dictionary with "fast", "full", "hyb" jet pT values.
    """
    bins, labels = get_bins(df, observable, 25)
    df["binned"] = pd.cut(df[observable], bins=bins, labels=labels)

    data = df[["binned", "wo_MatchProb", "w_MatchProb"]].copy()
    
    # Make routing decision for each event
    decision = get_decision(scores, logfile, **kwargs)
    
    # Expand decision from event-level to track-level
    #dec_expand = np.repeat(decision, df.groupby("EvtNumber").size().to_numpy())
    #data["decision"] = dec_expand

    #Associate each event-level decision explicitly with its EvtNumber
    decision_by_event = pd.Series(np.asarray(decision),index=score_event_ids)

    # Assign that event decision to every track belonging to the event
    row_event_ids = df.index.get_level_values("EvtNumber")

    data["decision"] = (decision_by_event.reindex(row_event_ids).to_numpy())

    # Compute hybrid MatchProb: use track overlay if decision=True, MC overlay otherwise
    data["h_MatchProb"] = np.where(
        data["decision"], data["wo_MatchProb"], data["w_MatchProb"]
    )

    if ret_diff:
        # Return jet pT differences for mismodelling analysis
        fast_pt = df["wo_leadJetpT"][:, 0]
        full_pt = df["w_leadJetpT"][:, 0]
        hyb_pt = np.where(data["decision"], df["wo_leadJetpT"], df["w_leadJetpT"])
        df["hybpt"] = hyb_pt
        hyb_pt = df["hybpt"][:, 0]
        
        pts = {"fast": fast_pt, "full": full_pt, "hyb": hyb_pt}
        return pts

    else:
        # Prepare data for binned efficiency calculation
        data = data.reset_index(0).set_index("binned").sort_index()
        data = data.loc[data.index.notnull()]
        data = data.set_index(data.groupby("binned").cumcount(), append=True)

        if observable == "wo_leadJetpT":
            data = data.drop_duplicates("EvtNumber")

        data = data.to_xarray().to_array().values
        data = np.transpose(data, [1, 2, 0])
        data = np.array(data, dtype=float)
        data = np.ma.masked_invalid(data)

        # Compute efficiencies per bin
        # Efficiency = number of matched tracks / total tracks
        fast_eff = np.count_nonzero(data[:, :, 1] > 0.5, axis=1)
        full_eff = np.count_nonzero(data[:, :, 2] > 0.5, axis=1)
        hyb_eff = np.count_nonzero(data[:, :, 4] > 0.5, axis=1)
        tot = np.ma.count(data[:, :, 0], axis=1)

        # Fraction of events sent to track overlay per bin
        frac = data[:, :, 3].sum(axis=1)
        
        isf = {"fast": fast_eff, "full": full_eff, "hyb": hyb_eff, "tot": tot, "frac": frac}
        return isf, bins


def simulate_roulette(scores, num_estimates, **kwargs):
    """
    Simulate roulette decision multiple times to get average behavior.
    
    For hard cut: returns deterministic decision.
    For smart roulette: averages over multiple stochastic samples.
    
    Parameters
    ----------
    scores : np.ndarray
        Event-level roulette scores.
    num_estimates : int
        Number of simulations (only for smart roulette).
    **kwargs : dict
        Routing parameters including quantile and rouletter type.
    
    Returns
    -------
    avg_decision : np.ndarray
        Average decision across simulations (for hard cut: binary; for smart: probability).
    """
    quantile = kwargs["quantile"]
    r_type = kwargs["rouletter"]
    thres = np.quantile(scores, quantile)    
    
    if r_type.casefold() == "hard":
        # Deterministic: same decision every time
        avg_decision = np.ma.masked_array(scores < thres).data
    else:
        # Stochastic: average over multiple samples
        decision_record = np.empty(shape=(num_estimates, len(scores)))
        for i in range(num_estimates):
            decision_record[i, :] = get_decision(scores, **kwargs)
        avg_decision = decision_record.mean(axis=0)
    
    return avg_decision


# =============================================================================
# Main Evaluation Functions
# =============================================================================

def permutation_importance(model, X, y, feature_names, n_repeats=10):

    print("Feature importance")

    # Nominal performance
    nominal_scores = model.predict(X, batch_size=8192, verbose=0).squeeze()
    nominal_auc = roc_auc_score(y, nominal_scores)

    importances = []

    for i, feature in enumerate(feature_names):

        print("now run feature ",feature_names[i])

        drops = []

        for n in range(n_repeats):

            print("n_repeats ",n)

            X_perm = X.copy()

            # Shuffle only this feature
            np.random.shuffle(X_perm[:, i])

            perm_scores = model.predict(X_perm, batch_size=8192, verbose=0).squeeze()
            perm_auc = roc_auc_score(y, perm_scores)

            drops.append(nominal_auc - perm_auc)

        importances.append({
            "feature": feature,
            "importance": np.mean(drops),
            "std": np.std(drops)
        })

    importances.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return nominal_auc, importances

#----------------------------------------------------------------------------------------------------

def evaluate_event(df, model, use_densities, scalers, sv_dir, xscore, evtmix, track_threshold, logfile=None,**kwargs):
    """
    Perform event-level evaluation with ML threshold optimization.
    
    This function evaluates the ML model's ability to route events between track overlay
    (fast) and MC overlay (full simulation). The evaluation compares ML-based routing to
    random selection across multiple threshold values.
    
    **ML Threshold Optimization:**
    The threshold is optimized by comparing ML selections to random baseline. For a given
    proportion of events sent to MC overlay, a fraction metric measures how well the 
    selection improves efficiency:
    
        fraction = (fullEff - hybEff) / (fullEff - fastEff)
    
    Lower fraction values indicate better agreement between hybrid and MC overlay,
    demonstrating the advantage of ML-based selection. This ensures maximum events
    use track overlay while maintaining reconstruction accuracy.
    
    **Evaluation includes:**
    - Multiple threshold cuts: {0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8}
    - Observable binning: Pt, Eta (and optionally leadJetpT)
    - Comparison: ML routing vs. random baseline with uncertainty bands
    - Metrics: Efficiency plots, fast fraction plots, roulette diagnostics
    
    Parameters
    ----------
    df : pd.DataFrame
        Event dataframe with features and match probabilities.
    model : keras.Model
        Trained classifier model.
    use_densities : bool
        Whether density features are used.
    scalers : tuple
        Feature scalers (FSscaler, CSscaler).
    sv_dir : str
        Save directory for plots.
    xscore : str
        Type of xScore metric.
    evtmix : str
        Event mixing strategy.
    logfile : file, optional
        Log file for output.
    **kwargs : dict
        Additional routing parameters (rouletter, sampler, smoothing, etc.).
    """
    sv_dir = f"results/{sv_dir}/plots/"
    os.makedirs(sv_dir, exist_ok=True)
    
    print("--------------------> ETA <-------------------------")
    eta = df["Eta"].to_numpy()
    plot_eta(eta,sv_dir)
    
    # Unpack scalers tuple if needed
    if isinstance(scalers, tuple):
        FSscaler, _ = scalers  # Extract feature scaler, ignore conditional scaler
    else:
        FSscaler = scalers
    
    print("="*60, file=logfile, flush=True)
    print("EVENT-LEVEL EVALUATION", file=logfile, flush=True)
    print("="*60, file=logfile, flush=True)
    
    # Compute event-level xScores
    #calculate one score per event
    scores, score_event_ids = get_event_xscore(model,FSscaler,df,use_densities,logfile,track_threshold=track_threshold,
        xscore=xscore,evtmix=evtmix)
    
    print(f"Event xScores computed: {scores.shape[0]} events", file=logfile, flush=True)
    
    # ML threshold optimization: scan multiple threshold cuts
    # These thresholds control what fraction of events are sent to track overlay
    cuts = [0.85]
    roul_thres = [np.quantile(scores, cut) for cut in cuts]
    
    print(f"\nScanning {len(cuts)} threshold cuts:", file=logfile, flush=True)
    for i, (cut, thres) in enumerate(zip(cuts, roul_thres)):
        print(f"  Cut {i}: quantile={cut:.2f}, threshold={thres:.4f}", file=logfile, flush=True)
    
    # Observables to analyze
    observables = {
        "Pt": ["$P_t$", "pt", ", Particle [GeV]"],
        "Eta": ["$\eta$", "eta", ", Particle"],
    }
    
    # Plot roulette score distribution
    #plots the distribution of event xScores and presumably draws the chosen score threshold(s) on it
    plot_roulette_score(scores, roul_thres, dir=sv_dir, logfile=logfile)
    
    # Evaluate each threshold cut
    for quality, threshold in enumerate(cuts):
        print(f"\n--- Evaluating threshold cut {quality} (quantile={threshold}) ---", file=logfile, flush=True)
        
        # Generate random baseline for comparison
        # Using 100 random estimates for uncertainty bands (reduced from 10000 for memory efficiency)
        #instead of intelligent ML-based event selection, it constructs a random baseline with the same nominal fraction
        randomScores, random_event_ids = get_event_xscore(
            model, FSscaler, df, use_densities, logfile,track_threshold=track_threshold,
            xscore="random", evtmix=evtmix,
            frac=cuts[quality], estimator=100
        )
        
        kwargs["quantile"] = threshold
        
        # Analyze each observable
        for key, value in observables.items():
            print(f"  Analyzing observable: {key}", file=logfile, flush=True)
            
            # Compute binned efficiency for ML routing
            isf, bins = binned_efficiency(df, scores, score_event_ids, key, ret_diff=False, logfile=logfile, **kwargs)
            
            # Compute random baseline with uncertainty
            down, up, mean = random_FA_spread(df, randomScores, random_event_ids, key, 3)
            
            # Simulate average routing behavior, comment since it's never used
            #avg_dec = simulate_roulette(scores, 1000, **kwargs)
            
            # Plot efficiency comparison (ML vs random)
            plot_efficiencies(
                isf, bins, down, up, mean, value[0],
                dir=sv_dir, logfile=logfile, units=value[2],
                quality=f"cut_{quality}", sv_prefix=value[1]
            )
            
            # Plot fraction of events sent to track overlay
            plot_fast_fraction(
                isf, bins, value[0],
                dir=sv_dir, logfile=logfile, units=value[2],
                quality=f"cut_{quality}", sv_prefix=value[1]
            )
    
    print("\n" + "="*60, file=logfile, flush=True)
    print("Event-level evaluation complete", file=logfile, flush=True)
    print("="*60, file=logfile, flush=True)


def evaluate_performance(model, sv_dir, validation=None, feat=None, cond=None, labels=None,
    history=None, logfile=None, isTraining=False, track_threshold=None, track_quantile=None):
    """
    Evaluate track-level model performance.
    
    Computes basic classification metrics and generates diagnostic plots:
    - Training/validation loss curves
    - Training/validation accuracy curves  
    - Score distributions (signal vs background)
    - ROC curves
    
    Parameters
    ----------
    model : keras.Model
        Trained classifier model.
    sv_dir : str
        Save directory for plots.
    validation : object, optional
        Validation dataset bundle (legacy interface).
    feat : np.ndarray, optional
        Test features.
    cond : np.ndarray, optional
        Conditional features (unused in current implementation).
    labels : np.ndarray, optional
        Test labels.
    history : dict, optional
        Training history dictionary.
    logfile : file, optional
        Log file for output.
    isTraining : bool, optional
        Whether evaluation is during training (unused).
    """
    sv_dir = f"results/{sv_dir}/plots/"
    os.makedirs(sv_dir, exist_ok=True)

    print("\n" + "="*60, file=logfile, flush=True)
    print("TRACK-LEVEL MODEL EVALUATION", file=logfile, flush=True)
    print("="*60, file=logfile, flush=True)

    # Evaluate model on test set
    test_loss, test_acc = model.evaluate(feat, labels, verbose=2)
    
    print(f"\nTest Loss: {test_loss:.4f}", file=logfile, flush=True)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)", file=logfile, flush=True)
    
    # Get model predictions
    #NOTE THAT THESE ARE NOT LOSSES, They are the neural-network outputs — the sigmoid classifier scores
    losses = model.predict(feat)
    losses = losses.squeeze()
    
    # Generate diagnostic plots
    if history is not None:
        plot_losses(history, sv_dir, logfile)
        plot_accuracy(history, sv_dir, logfile)
    
    plot_scores(losses, labels, sv_dir, logfile,track_threshold=track_threshold,track_quantile=track_quantile)
    plot_ROC(losses, labels, sv_dir, logfile)
    
    print("="*60, file=logfile, flush=True)
    print("Track-level evaluation complete", file=logfile, flush=True)
    print("="*60 + "\n", file=logfile, flush=True)
