import sys
import os
# Add parent directory to path so we can import from data/, network/, utils/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import datetime
import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

from data.dataloader_multi import loadEvaluationData
from utils.evaluator_multi import evaluate_performance, evaluate_event, permutation_importance
from utils.args import get_args
from utils.plotting import plot_features

import glob
import pandas as pd

def main():
    args = get_args()
    
     # Set random seed for reproducible evaluation
    np.random.seed(args.seed)
    
    datetimenow = datetime.datetime.now().strftime("%B %d, %Y: %H:%M:%S")
    
    # Data/eval config
    sample = args.sample  # Sample used for training (for loading model)
    eval_sample = args.eval_sample if args.eval_sample else args.sample  # Sample to evaluate on
    xscore = args.xscore
    use_density = bool(args.usedensity)
    roul_type = args.rouletter
    temperature = args.temperature
    path = 'data'
    
    sv_dir = sample + "/" + xscore + "/" + roul_type
    if xscore[:7] == "density":
        sv_dir += "/" + args.evtmix
    if (roul_type.casefold() == 'smart') and temperature is not None:
        sv_dir = sv_dir + "/" + str(temperature)
    
    log_sv_dir = f"results/{sv_dir}/logs/"
    os.makedirs(log_sv_dir, exist_ok=True)
    
    # Load evaluation data (no train/test split needed - use entire dataset)
    path = "data"
    
    print(f"Loading evaluation data for sample: {eval_sample}")
    if eval_sample != sample:
        print(f"  Note: Evaluating on different sample than training ({sample} → {eval_sample})")
    

    matched_file = glob.glob(f"data/matched_{eval_sample}_data*.h5")[0]
    unmatched_file = glob.glob(f"data/unmatched_{eval_sample}_data*.h5")[0]

    df_matched = pd.read_hdf(matched_file)
    df_unmatched = pd.read_hdf(unmatched_file)

    df_full = pd.concat([df_matched, df_unmatched], axis=0 )

    print("-----------CHECK STARTS --------------------------")
    mc_only = df_full[
        (df_full["w_hasMatch"] == True) &
        (df_full["wo_hasMatch"] == False)
    ]

    track_only = df_full[
        (df_full["w_hasMatch"] == False) &
        (df_full["wo_hasMatch"] == True)
    ]

    print("MC-only Diff:", len(mc_only))
    print("Track-only Diff:", len(track_only))

    print("Maximum E, MC-only:", mc_only["E"].max())
    print("Maximum E, Track-only:", track_only["E"].max())

    print("Full unsampled dataframe:", df_full.shape)

    diff = df_full[df_full["TargetLabel"] == 1]["E"]
    nodiff = df_full[df_full["TargetLabel"] == 0]["E"]

    print("Number NoDiff:", len(nodiff))
    print("Number Diff:", len(diff))

    print("Maximum E for NoDiff:", nodiff.max())
    print("Maximum E for Diff:", diff.max())
    
    #classifier path
    cl_dir = f"results/{sample}/classifier"
    
    threshold_path = f"{cl_dir}/track_threshold.pkl"
    with open(threshold_path, "rb") as f:
        threshold_info = pickle.load(f)
    track_threshold = threshold_info["threshold"]
    track_quantile = threshold_info["quantile"]

    print(
        f"Using fixed track threshold: {track_threshold:.6f} "
        f"(derived from validation {track_quantile:.0%} quantile)" )
    
    # Decide which events to evaluate
    event_ids = None
    
    if eval_sample == sample:
        split_path = f"{cl_dir}/split_events.pkl"
        with open(split_path, "rb") as f:
            split_events = pickle.load(f)
        event_ids = split_events["test"]
        print(f"Using {len(event_ids)} test events")
    else:
        print(f"Evaluating independent sample {eval_sample}; using all events")
    
    # You can specify sample sizes to limit evaluation, or None to use all data
    F_eval, L_eval, feat_list, df_eval = loadEvaluationData(
        path=path,logfile=None,sample=eval_sample,use_densities=use_density,event_ids=event_ids,event_sample_size=None )
    
    #F_eval: numerical ML feature array
    #L_eval: TargetLabel values
    #feat_list: names of the features
    #df_eval: original/full Pandas DataFrame, needed for the later event-level evaluation -> needs the complete dataframe
    
    print(f"Evaluation data loaded: {F_eval.shape}")
    
    plot_features(df_eval, f"results/{sv_dir}/plots")
    
    # Load trained model
    model_path = f"{cl_dir}/classifier.h5"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            f"Please run train_model.py first."
        )
    
    #sample identifies which trained classifier to load, while eval_sample identifies which dataset to test it on
    
    print(f"Loading model from {model_path}")
    trkPredictor = keras.models.load_model(model_path)
    
    # Load the scaler
    scaler_path = f"{cl_dir}/scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    F_eval = scaler.transform(F_eval)
    
    # Load training history if available
    history_path = f"{cl_dir}/history.pkl"
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)
    else:
        history = None
        print(f"Warning: Training history not found at {history_path}")
    
    # Logging
    logfile = open(f"{log_sv_dir}/eval_log.txt", "w")
    print("\n\n", file=logfile)
    print("=*" * 60, file=logfile, flush=True)
    print("*=" * 60, file=logfile, flush=True)
    print(f"\nEvaluation started: {datetimenow}", file=logfile, flush=True)
    print(f"Evaluation data shape: {F_eval.shape}", file=logfile, flush=True)
    print(f"Model loaded from: {model_path}", file=logfile, flush=True)
    
    # Track-level evaluation
    print("\n=== Track-level evaluation ===", file=logfile, flush=True)
    evaluate_performance(
        trkPredictor,
        sv_dir,
        feat=F_eval,
        labels=L_eval,
        history=history,
        logfile=logfile,
        isTraining=True,
        track_threshold=track_threshold,
        track_quantile=track_quantile
    )
    
    # Feature importance --- (SLOW!)
    print("\n=== Feature importance ===", file=logfile, flush=True)

    nominal_auc, importances = permutation_importance(trkPredictor,F_eval,L_eval,feat_list,n_repeats=10)
    print(f"Nominal AUC: {nominal_auc:.4f}", file=logfile)

    for item in importances:
        print(f"{item['feature']:20s} "f"{item['importance']:.5f} +/- {item['std']:.5f}", file=logfile)
    #--
    
    # Event-level evaluation
    print("\n=== Event-level evaluation ===", file=logfile, flush=True)
    
    # Use the dataframe we already loaded
    print(f"Using loaded dataframe with {len(df_eval)} samples", file=logfile, flush=True)
    
    dec_kwargs = {
        "smoothing": temperature,
        "rouletter": roul_type
    }
    
    evaluate_event(
        df_eval,
        trkPredictor,
        use_density,
        scaler,
        sv_dir,
        xscore,
        args.evtmix,
        track_threshold,
        logfile,
        **dec_kwargs
    )
    
    print("\n✓ Evaluation complete!", file=logfile, flush=True)
    print("="*60, file=logfile, flush=True)
    
    logfile.close()
    
    # Also print to console
    print(f"✓ Evaluation complete!")
    print(f"  Results saved to: results/{sv_dir}")
    print(f"  Logs saved to: {log_sv_dir}")


if __name__ == "__main__":
    main()
