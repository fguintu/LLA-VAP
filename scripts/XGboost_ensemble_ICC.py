import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import json
from pathlib import Path
import seaborn as sns
from typing import Dict, Tuple, List, Any
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fft import fft


def extract_advanced_features(data: np.ndarray, window_sizes=[5, 10, 20]) -> np.ndarray:
    """Extract comprehensive temporal and frequency domain features"""
    # Convert boolean to float if needed
    if data.dtype == bool:
        data = data.astype(float)

    features_list = []

    for window_size in window_sizes:
        # Basic temporal features
        rolling_mean = np.convolve(data, np.ones(window_size) / window_size, mode='same')
        rolling_std = pd.Series(data).rolling(window_size).std().fillna(0).values

        # Gradient features
        gradient = np.gradient(data)
        gradient_mean = np.convolve(gradient, np.ones(window_size) / window_size, mode='same')

        # Peak detection
        peaks, _ = find_peaks(data, distance=window_size)
        valleys, _ = find_peaks(-data, distance=window_size)
        peak_indicator = np.zeros_like(data)
        valley_indicator = np.zeros_like(data)
        peak_indicator[peaks] = 1
        valley_indicator[valleys] = 1

        # Skip change point detection
        change_point_indicator = np.zeros_like(data)

        # Frequency domain features
        n_fft = min(window_size, len(data))
        fft_features = np.abs(fft(data))[:n_fft]
        # Pad FFT features to match data length
        fft_features = np.pad(fft_features, (0, len(data) - len(fft_features)), 'constant')

        # Statistical features
        skewness = pd.Series(data).rolling(window_size).apply(skew, raw=True).fillna(0).values
        kurt = pd.Series(data).rolling(window_size).apply(kurtosis, raw=True).fillna(0).values

        all_features = [
            rolling_mean,
            rolling_std,
            gradient,
            gradient_mean,
            peak_indicator,
            valley_indicator,
            change_point_indicator,
            skewness,
            kurt,
            fft_features
        ]

        features_list.extend([np.array(f).reshape(len(data)) for f in all_features])

    return np.column_stack(features_list)


def create_enhanced_features(llama_results: Dict, vap_results: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create comprehensive feature set using both probabilities and predictions"""
    X = []
    y = []
    file_mapping = []

    vap_data = vap_results['per_stimulus_results']

    for stim_key in llama_results.keys():
        if stim_key not in vap_data:
            continue

        print(f"Processing stimulus: {stim_key}")
        llama_data = llama_results[stim_key]
        vap_stimulus_data = vap_data[stim_key]

        # Get all signals
        llama_probs = np.array(llama_data['probabilities'])
        llama_preds = (llama_probs > 0.5).astype(float)
        vap_probs = np.array(vap_stimulus_data['probabilities'])
        vap_preds = np.array(vap_stimulus_data['predictions']).astype(float)
        ground_truth = np.array(llama_data['ground_truth'])

        min_len = min(len(llama_probs), len(vap_preds), len(ground_truth))
        print(f"Sequence length: {min_len}")

        try:
            # Extract features for both probability signals
            llama_prob_features = extract_advanced_features(llama_probs[:min_len])
            vap_prob_features = extract_advanced_features(vap_probs[:min_len])

            # Combine all features
            features = np.column_stack([
                # Raw signals
                llama_probs[:min_len],
                llama_preds[:min_len],
                vap_probs[:min_len],
                vap_preds[:min_len],
                # Advanced features from probabilities
                llama_prob_features,
                vap_prob_features
            ])

            # Remove any remaining inf or nan values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

            X.append(features)
            y.append(ground_truth[:min_len])
            file_mapping.extend([stim_key] * min_len)

            print(f"Features shape for {stim_key}: {features.shape}")

        except Exception as e:
            import traceback
            print(f"Error processing {stim_key}: {str(e)}")
            print(traceback.format_exc())
            continue

    if not X:
        raise ValueError("No features were successfully extracted")

    return np.vstack(X), np.concatenate(y), np.array(file_mapping)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: np.ndarray, window_size: int = 75) -> Dict:
    """Comprehensive evaluation including window-based metrics"""
    metrics = {}

    # Create windows around ground truth positives
    true_windows = np.zeros_like(y_true)
    for i in np.where(y_true == 1)[0]:
        start = max(0, i - window_size)
        end = min(len(y_true), i + window_size + 1)
        true_windows[start:end] = 1

    # Calculate basic metrics
    tn, fp, fn, tp = confusion_matrix(true_windows, y_pred).ravel()

    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'balanced_accuracy': balanced_accuracy_score(true_windows, y_pred),
        'precision': precision_score(true_windows, y_pred),
        'recall': recall_score(true_windows, y_pred),
        'f1': f1_score(true_windows, y_pred),
        'window_size': window_size
    })

    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(true_windows, y_prob)
    precision, recall, _ = precision_recall_curve(true_windows, y_prob)

    metrics.update({
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': auc(fpr, tpr)
        },
        'pr_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'auc': auc(recall, precision)
        }
    })

    return metrics


def train_ensemble_kfold(X: np.ndarray, y: np.ndarray,
                         file_mapping: np.ndarray, output_dir: Path,
                         n_splits: int = 4) -> Dict:
    """Train enhanced ensemble with optimized parameters"""
    unique_files = np.unique(file_mapping)
    kf = KFold(n_splits=min(n_splits, len(unique_files)), shuffle=True, random_state=42)
    scaler = StandardScaler()

    results = {
        'fold_metrics': [],
        'feature_importances': []
    }

    # Calculate class weights
    scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
    print(f"\nClass imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

    # Setup plots
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 8))

    feature_names = ['llama_prob', 'llama_pred', 'vap_prob', 'vap_pred']
    for i in range((X.shape[1] - 4) // 2):  # Additional feature names for each model
        feature_names.extend([f'llama_feature_{i}', f'vap_feature_{i}'])

    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_files)):
        print(f"\nTraining fold {fold + 1}/{min(n_splits, len(unique_files))}")

        train_files = unique_files[train_idx]
        test_files = unique_files[test_idx]

        train_mask = np.isin(file_mapping, train_files)
        test_mask = np.isin(file_mapping, test_files)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train XGBoost model with optimized parameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'aucpr'],
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'min_child_weight': 3,
            'gamma': 0.1
        }

        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_names)

        # Train model
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=200,
            early_stopping_rounds=20,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            verbose_eval=False
        )

        # Make predictions
        y_prob = model.predict(dtest)

        # Find optimal threshold
        thresholds = np.linspace(0.1, 0.9, 100)
        best_thresh = 0.5
        best_bacc = 0

        for thresh in thresholds:
            y_pred = (y_prob > thresh).astype(int)
            bacc = balanced_accuracy_score(y_test, y_pred)
            if bacc > best_bacc:
                best_bacc = bacc
                best_thresh = thresh

        # Final predictions using best threshold
        y_pred = (y_prob > best_thresh).astype(int)

        # Evaluate predictions
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics.update({
            'fold': fold,
            'optimal_threshold': float(best_thresh),
            'train_files': train_files.tolist(),
            'test_files': test_files.tolist()
        })

        # Plot ROC and PR curves
        ax_roc.plot(metrics['roc_curve']['fpr'],
                    metrics['roc_curve']['tpr'],
                    label=f'Fold {fold} (AUC = {metrics["roc_curve"]["auc"]:.2f})')

        ax_pr.plot(metrics['pr_curve']['recall'],
                   metrics['pr_curve']['precision'],
                   label=f'Fold {fold} (AUC = {metrics["pr_curve"]["auc"]:.2f})')

        results['fold_metrics'].append(metrics)

        # Store feature importance
        importance_dict = {name: float(score) for name, score in
                           model.get_score(importance_type='gain').items()}
        results['feature_importances'].append(importance_dict)

        print(f"Fold {fold + 1} metrics:")
        print(f"Optimal threshold: {best_thresh:.3f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")
        print(f"AUC-ROC: {metrics['roc_curve']['auc']:.3f}")

    # Finalize plots
    for fig, ax, title in [(fig_roc, ax_roc, 'ROC Curves'),
                           (fig_pr, ax_pr, 'Precision-Recall Curves')]:
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

        if title == 'ROC Curves':
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
        else:
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')

        fig.savefig(output_dir / f"{title.lower().replace(' ', '_')}.png")
        plt.close(fig)

    # Plot feature importance
    importance_df = pd.DataFrame(results['feature_importances'])
    importance_df = importance_df.fillna(0)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.melt(importance_df), x='variable', y='value')
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()

    return results


def main():
    # Setup paths
    llama_results_path = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_llm_ICC_output/3b/0.6_no_flip/llama_realtime_results.json")
    vap_results_path = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_VAP_ICC_output/0.999_no_flip_best/full_results.json")
    ensemble_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/enhanced_xgb_ensemble_ICC_output")
    ensemble_dir.mkdir(exist_ok=True)

    print("\nLoading result files...")
    with open(llama_results_path) as f:
        llama_results = json.load(f)
    with open(vap_results_path) as f:
        vap_results = json.load(f)

    print("\nCreating enhanced features...")
    X, y, file_mapping = create_enhanced_features(llama_results, vap_results)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of unique files: {len(np.unique(file_mapping))}")
    print(f"Class distribution: {np.bincount(y)}")

    print("\nTraining ensemble model...")
    results = train_ensemble_kfold(X, y, file_mapping, ensemble_dir)

    # Calculate average metrics across folds
    metrics_df = pd.DataFrame([m for m in results['fold_metrics']])
    mean_metrics = metrics_df[[
        'accuracy', 'balanced_accuracy', 'precision',
        'recall', 'f1'
    ]].mean()
    std_metrics = metrics_df[[
        'accuracy', 'balanced_accuracy', 'precision',
        'recall', 'f1'
    ]].std()

    # Calculate average feature importance
    importance_df = pd.DataFrame(results['feature_importances'])
    mean_importance = importance_df.mean()
    std_importance = importance_df.std()

    # Prepare final results
    final_results = {
        'average_metrics': {
            metric: {
                'mean': float(mean_metrics[metric]),
                'std': float(std_metrics[metric])
            }
            for metric in mean_metrics.index
        },
        'feature_importance': {
            feature: {
                'mean': float(mean_importance.get(feature, 0)),
                'std': float(std_importance.get(feature, 0))
            }
            for feature in importance_df.columns
        },
        'fold_metrics': results['fold_metrics'],
        'training_stats': {
            'feature_matrix_shape': X.shape,
            'total_samples': len(y),
            'positive_samples': int(np.sum(y == 1)),
            'negative_samples': int(np.sum(y == 0)),
            'class_ratio': float(np.sum(y == 0) / np.sum(y == 1))
        }
    }

    # Print summary
    print("\nAverage Metrics (mean ± std):")
    for metric, values in final_results['average_metrics'].items():
        print(f"{metric}: {values['mean']:.3f} ± {values['std']:.3f}")

    print("\nTop Feature Importance (mean ± std):")
    importance_items = sorted(final_results['feature_importance'].items(),
                              key=lambda x: x[1]['mean'],
                              reverse=True)
    for feature, values in importance_items[:10]:  # Top 10 features
        print(f"{feature}: {values['mean']:.3f} ± {values['std']:.3f}")

    # Save detailed results
    with open(ensemble_dir / "ensemble_results.json", 'w') as f:
        json.dump(final_results, f, indent=4)

    # Save metrics to CSV
    metrics_df.to_csv(ensemble_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(final_results['average_metrics']).T.to_csv(
        ensemble_dir / "average_metrics.csv"
    )

    # Save feature importance to CSV
    importance_df = pd.DataFrame(final_results['feature_importance']).T
    importance_df.columns = ['mean', 'std']
    importance_df.sort_values('mean', ascending=False, inplace=True)
    importance_df.to_csv(ensemble_dir / "feature_importance.csv")

    print(f"\nAll results saved to {ensemble_dir}")
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error in execution: {str(e)}")
        print(traceback.format_exc())