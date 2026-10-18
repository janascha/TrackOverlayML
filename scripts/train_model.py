import sys
import os
# Add parent directory to path so we can import from data/, network/, utils/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import datetime
import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam #optimizer: controls how the neural-network weights are updated
from tensorflow.keras.losses import binary_crossentropy #loss function for your two-class problem.
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, LearningRateScheduler

from network.classifier import getTrkPredictor
from data.dataloader_multi import loadTrainValTestData
from utils.args import get_args
from utils.io import save_settings


def main():
    args = get_args()
    
    datetimenow = datetime.datetime.now().strftime("%B %d, %Y: %H:%M:%S")
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Hyperparameters
    #Read the neural-network hyperparameters
    layers = args.layers
    lr = args.lr
    nepochs = args.epochs
    batchsize = args.batchsize
    loss = binary_crossentropy
    optim = Adam(learning_rate=lr)
    
    # Data/eval config
    #Read some evaluation/routing configuration -> normally relevant only for the evaluation...
    sample = args.sample
    xscore = args.xscore
    roul_type = args.rouletter
    temperature = args.temperature
    
    sv_dir = sample + "/" + xscore + "/" + roul_type
    if xscore[:7] == "density":
        sv_dir += "/" + args.evtmix
    if (roul_type.casefold() == 'smart') and temperature is not None:
        sv_dir = sv_dir + "/" + str(temperature)
    
    log_sv_dir = f"results/{sv_dir}/logs/"
    os.makedirs(log_sv_dir, exist_ok=True)
    
    # Load processed data and create train/test split
    path = "data"
    sample = args.sample
    split = args.trainsplit
    use_density = bool(args.usedensity)
    
    print(f"Loading processed data for sample: {sample}")
    
    # Optional: Use subset for balanced training
    matched_size = args.matched_size if hasattr(args, 'matched_size') else None
    unmatched_size = args.unmatched_size if hasattr(args, 'unmatched_size') else None
    
    if matched_size or unmatched_size:
        print(f"  Training on SUBSET:")
        if matched_size:
            print(f"    Matched samples: {matched_size}")
        if unmatched_size:
            print(f"    Unmatched samples: {unmatched_size}")

    #Load the processed data ------------------------------------------------------------------
    #This is where the prepared HDF5 files are turned into the arrays that actually enter training.
    (F_train,F_val,F_test,L_train,L_val,L_test,W_train,W_val,W_test,feat_list,scaler,split_events) = loadTrainValTestData(
        path=path,
        logfile=None,  # Will log to file below
        sample=sample,
        split=split,
        use_densities=use_density,
        event_sample_size=None
    )
    
    numFeatures = len(feat_list)
    
    # Logging
    logfile = open(f"{log_sv_dir}/training_log.txt", "w")
    print("\n\n", file=logfile)
    print("=*" * 60, file=logfile, flush=True)
    print("*=" * 60, file=logfile, flush=True)
    print(f"\nTraining started: {datetimenow}", file=logfile, flush=True)
    print(f"Training data shape: {F_train.shape}", file=logfile, flush=True)
    print(f"Number of features: {numFeatures}", file=logfile, flush=True)
    print(f"Feature list: {feat_list}", file=logfile, flush=True)
    
    # Save settings
    save_settings(sv_dir, args)
    
    # Build model
    print(
        "\nInitialising and training classifier from scratch.",
        file=logfile,
        flush=True,
    )
    
    #this creates the Keras model object (network/classifyer.py)
    trkPredictor = getTrkPredictor(numFeatures, layers)
    
    #architecture becomes configured for actual training
    #this tells Keras how that network should be trained
    trkPredictor.compile(loss=loss, optimizer=optim, metrics=["accuracy"])
    
    #prints a Keras summary of the architecture into the training log, including the layer names, output shapes, and number of trainable parameters
    print(trkPredictor.summary(), file=logfile, flush=True)
    
    # Callbacks ----------------------------------------------------------------------------------------------------
    
    #After an epoch, look at the validation loss. If this is the best validation loss seen so far, save the model.
    checkpoint = ModelCheckpoint(
        './results/saved_models/model-{epoch:03d}.ckpt',
        monitor='val_loss',
        verbose=2,
        save_best_only=True,
        mode='min'
    )
    #writes the training history after every epoch into a CSV file
    csvLogger = CSVLogger(f"{log_sv_dir}/training_logger.csv", separator=",", append=False)
    
    #Stop training if val_loss has not improved for 20 epochs.
    earlyStopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        verbose=1,
        restore_best_weights=True
    )
    
    #custom learning-rate schedule
    #epochs 0 and 1: keep the learning rate unchanged
    #from epoch 2 onward: multiply the current learning rate by \(e^{-0.1}\) each epoch
    def scheduler(epoch, lr):
        if epoch < 2:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    
    learningrate = LearningRateScheduler(scheduler)
    
    #all callbacks are collected
    callback = [csvLogger, earlyStopping, checkpoint, learningrate]
    # <- Callbacks -----------------------------------------------------------------------------------------
    
    
    # Train
    print("Training Begins", file=logfile, flush=True)
    train_start = time.time()
    
    history = trkPredictor.fit(
        F_train,   #F_train contains the input features
        L_train,   #L_train contains the corresponding target labels, 0 or 1
        epochs=nepochs, #sets the maximum number of epochs.
        sample_weight=W_train, #class-balancing weights
        #validation_data=(F_test, L_test, W_test), #after each epoch, Keras also evaluates the model on the test/validation sample. validation loss is also weighted
        validation_data=(F_val, L_val, W_val),
        batch_size=batchsize, #how many training samples are processed before one gradient update. After all training samples have been processed once, one epoch is complete.
        callbacks=callback, #CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
    )
    #F_train -> training set
    #F_test  -> validation set during training
    
    train_end = time.time()
    #save the history as a normal Python dictionary, with one entry per epoch
    history = history.history
    
    print(
        f"Finished training in {str(datetime.timedelta(seconds=round(train_end-train_start)))}",
        file=logfile,
        flush=True
    )
    
    # Save model and history
    cl_dir = f"results/{sample}/classifier"
    os.makedirs(cl_dir, exist_ok=True)
    
    track_quantile = 0.58
    val_scores = trkPredictor.predict(F_val,batch_size=8192,verbose=0).squeeze()
    track_threshold = np.quantile(val_scores[L_val == 1.0],track_quantile)
    print(
        f"Track threshold from validation sample: "
        f"quantile={track_quantile}, "
        f"threshold={track_threshold:.6f}",
        file=logfile,
        flush=True)

    # Save threshold
    threshold_info = {"quantile": track_quantile,"threshold": float(track_threshold)}
    with open(f"{cl_dir}/track_threshold.pkl", "wb") as f:
        pickle.dump(threshold_info, f)

    #save the scaler:
    with open(f"{cl_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    with open(f"{cl_dir}/split_events.pkl", "wb") as f:
        pickle.dump(split_events, f)
    
    #the model itself is saved
    #That .h5 file contains the trained network parameters and model configuration needed to reload the classifier later
    trkPredictor.save(f"{cl_dir}/classifier.h5")
    
    #save Python history dictionary in a separate file
    with open(f"{cl_dir}/history.pkl", "wb") as file_pi:
        pickle.dump(history, file_pi)
    
    print(f"\n✓ Model saved to {cl_dir}/classifier.h5", file=logfile, flush=True)
    print(f"✓ Training history saved to {cl_dir}/history.pkl", file=logfile, flush=True)
    print("="*60, file=logfile, flush=True)
    
    logfile.close()
    
    # Also print to console
    print(f"✓ Training complete!")
    print(f"  Training time: {str(datetime.timedelta(seconds=round(train_end-train_start)))}")
    print(f"  Model saved to: {cl_dir}/classifier.h5")
    print(f"  Logs saved to: {log_sv_dir}")


if __name__ == "__main__":
    main()
