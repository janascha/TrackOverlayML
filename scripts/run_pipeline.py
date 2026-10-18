import sys
import os
# Add parent directory to path so we can import from data/, network/, utils/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import argparse


def run_step(script_path, args_list, step_name):
    """Run a script with given arguments."""
    cmd = [sys.executable, script_path] + args_list
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"Running: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n✗ {step_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n✓ {step_name} completed successfully")
    print('='*60)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline orchestrator for TrackOverlayML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_pipeline.py --stage all --sample ttbar
  
  # Run only data preparation
  python scripts/run_pipeline.py --stage data --sample ttbar
  
  # Run training and evaluation
  python scripts/run_pipeline.py --stage train --sample ttbar
  python scripts/run_pipeline.py --stage eval --sample ttbar
        """
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "data", "train", "eval"],
        default="all",
        help="Pipeline stage to run: all (default), data, train, or eval"
    )
    
    # Pass through arguments to the scripts (matches args.py definitions)
    parser.add_argument("--path", type=str, default="data", help="Data directory path")
    parser.add_argument("--sample", type=str, default="JZ7W", help="Sample name")
    parser.add_argument("--seed", type=int, default=23, help="Random seed")
    parser.add_argument("--trainsplit", type=float, default=0.7, help="Train/test split")
    parser.add_argument("--layers", nargs="*", type=int, default=[45, 35, 30], help="Layer sizes")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batchsize", type=int, default=80, help="Batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--rouletter", type=str, default="smart", help="Roulette type")
    parser.add_argument("--temperature", type=float, default=0.00005, help="Temperature for smart roulette")
    parser.add_argument("--xscore", type=str, default="simpleratio", help="XScore definition")
    parser.add_argument("--evtmix", type=str, default="None", help="Event mix type")
    parser.add_argument("--usedensity", type=int, default=1, help="Use density features")
    parser.add_argument("--eval_sample", type=str, default=None, help="Sample for evaluation (if different)")
    
    args = parser.parse_args()
    
    # Get the scripts directory
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pipeline stages with stage-specific arguments
    if args.stage in ["all", "data"]:
        prepare_script = os.path.join(scripts_dir, "prepare_data.py")
        # prepare_data.py only needs: path, sample, seed, trainsplit, usedensity
        prepare_args = [
            f"--path={args.path}",
            f"-sn={args.sample}",
            f"-s={args.seed}",
            f"--trainsplit={args.trainsplit}",
            f"-ud={args.usedensity}",
        ]
        run_step(prepare_script, prepare_args, "Data Preparation")
    
    if args.stage in ["all", "train"]:
        train_script = os.path.join(scripts_dir, "train_model.py")
        # train_model.py needs all training parameters
        train_args = [
            f"--path={args.path}",
            f"-sn={args.sample}",
            f"-s={args.seed}",
            f"--trainsplit={args.trainsplit}",
            f"-ud={args.usedensity}",
            f"--epochs={args.epochs}",
            f"-bs={args.batchsize}",
            f"-lr={args.lr}",
            f"--layers"] + [str(l) for l in args.layers] + [
            f"-p={args.patience}",
            f"-xs={args.xscore}",
            f"-rltr={args.rouletter}",
            f"-temp={args.temperature}",
            f"-em={args.evtmix}",
        ]
        if args.matched_size is not None:
            train_args.append(f"--matched_size={args.matched_size}")
        if args.unmatched_size is not None:
            train_args.append(f"--unmatched_size={args.unmatched_size}")
        run_step(train_script, train_args, "Model Training")
    
    if args.stage in ["all", "eval"]:
        eval_script = os.path.join(scripts_dir, "evaluate_model.py")
        # evaluate_model.py needs evaluation parameters
        eval_args = [
            f"--path={args.path}",
            f"-sn={args.sample}",
            f"-s={args.seed}",
            f"-ud={args.usedensity}",
            f"-xs={args.xscore}",
            f"-rltr={args.rouletter}",
            f"-temp={args.temperature}",
            f"-em={args.evtmix}",
        ]
        if args.eval_sample is not None:
            eval_args.append(f"--eval_sample={args.eval_sample}")
        if args.matched_size is not None:
            eval_args.append(f"--matched_size={args.matched_size}")
        if args.unmatched_size is not None:
            eval_args.append(f"--unmatched_size={args.unmatched_size}")
        run_step(eval_script, eval_args, "Model Evaluation")
    
    print("\n" + "="*60)
    print("="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    print("="*60)


if __name__ == "__main__":
    main()
