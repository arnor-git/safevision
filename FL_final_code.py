import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import random
import json
import time
from itertools import cycle

ADMIN_PASSWORD = "arj123"
NUM_CLIENTS = 2
NUM_FOLDS = 4
NUM_EPOCHS = 30
NUM_ROUNDS = 5

DROPOUT_RATE = 0.2
L2_REG = 0.001
LEARNING_RATE = 0.001

DUMMY_STUDENT_NAMES = [
    "aladdin123", "jasmine456", "simba789", "nala321", "mulan654",
    "pocahontas987", "tarzan246", "ariel135", "belle579", "beast864",
    "cinderella753", "aurora951", "phillip246", "snow357", "rapunzel468",
    "merida159", "elsa753", "anna246", "olaf135", "tiana864"
]

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('secure_data', exist_ok=True)
os.makedirs('client_data', exist_ok=True)


class SimpleIDProtector:
    def __init__(self):
        self.original_to_dummy = {}
        self.dummy_to_original = {}

    def generate_dummy_student_ids(self, orig_ids):
        unique_ids = np.unique(orig_ids)
        dummy_names = DUMMY_STUDENT_NAMES.copy()
        random.shuffle(dummy_names)

        for idx, original_id in enumerate(unique_ids):
            if idx < len(dummy_names):
                dummy_student_id = dummy_names[idx]
            else:
                dummy_student_id = f"user{idx + 1000}"

            self.original_to_dummy[original_id] = dummy_student_id
            self.dummy_to_original[dummy_student_id] = original_id

        dummy_student_ids = np.array([self.original_to_dummy[id] for id in orig_ids])
        self.save_mapping()
        return dummy_student_ids

    def save_mapping(self):
        mapping_data = {
            "original_to_dummy": {str(k): str(v) for k, v in self.original_to_dummy.items()},
            "dummy_to_original": {str(k): int(v) for k, v in self.dummy_to_original.items()},
            "admin_password": ADMIN_PASSWORD
        }

        with open('secure_data/std_id_mapping.json', 'w') as f:
            json.dump(mapping_data, f)

        dummy_metadata = {
            'num_students': len(self.dummy_to_original),
            'dummy_names': list(self.dummy_to_original.keys())
        }
        with open('secure_data/dummy_student_id_info.json', 'w') as f:
            json.dump(dummy_metadata, f, indent=2)

    @classmethod
    def load_mapping(cls):
        try:
            with open('secure_data/std_id_mapping.json', 'r') as f:
                mapping_data = json.load(f)

            protector = cls()
            protector.original_to_dummy = {int(k): str(v) for k, v in mapping_data["original_to_dummy"].items()}
            protector.dummy_to_original = {str(k): int(v) for k, v in mapping_data["dummy_to_original"].items()}
            return protector
        except Exception as e:
            print(f"Error loading mapping: {e}")
            return None

    def dummy_to_original_id(self, dummy_student_id):
        return self.dummy_to_original.get(dummy_student_id)


def dataprepration(data_path, test_size=0.2, num_clients=NUM_CLIENTS, random_state=42):
    print(f"Loading data from {data_path}...")

    studentidpredictiondf = pd.read_csv(data_path)

    # Remove unnamed columns
    unnamed_cols = [col for col in studentidpredictiondf.columns if 'Unnamed' in col]
    if unnamed_cols:
        studentidpredictiondf = studentidpredictiondf.drop(columns=unnamed_cols)

    print(f"Dataset: {studentidpredictiondf.shape[0]} rows, {studentidpredictiondf.shape[1]} columns")
    print(f"Unique students: {studentidpredictiondf['student_id'].nunique()}")

    # Create dummy ID mappings
    original_student_ids = studentidpredictiondf['student_id'].values
    id_protector = SimpleIDProtector()
    dummy_student_ids = id_protector.generate_dummy_student_ids(original_student_ids)

    studentidpredictiondf['student_id'] = dummy_student_ids

    # Random stratified splitting
    print(f"\n🎯 USING RANDOM STRATIFIED SPLITTING")

    # Drop unnecessary columns
    df_processed = studentidpredictiondf.drop(columns=['game_level'], errors='ignore')

    X = df_processed.drop(columns=['student_id'])
    y_student_ids = df_processed['student_id']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_student_ids)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training samples: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"Testing samples: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Save preprocessing objects
    joblib.dump(scaler, 'models/student_id_feature_scaler.pkl')
    joblib.dump(label_encoder, 'models/student_id_dummy_encoder.pkl')

    # Split training data among federated clients
    client_data = []

    for client_idx in range(num_clients):
        if client_idx == num_clients - 1:
            start_idx = client_idx * (len(X_train) // num_clients)
            client_X = X_train_scaled.iloc[start_idx:].copy()
            client_y = y_train[start_idx:]
        else:
            start_idx = client_idx * (len(X_train) // num_clients)
            end_idx = (client_idx + 1) * (len(X_train) // num_clients)
            client_X = X_train_scaled.iloc[start_idx:end_idx].copy()
            client_y = y_train[start_idx:end_idx]

        # Calculate class weights
        unique_classes_client = np.unique(client_y)
        class_counts = np.bincount(client_y)
        total_samples = len(client_y)
        class_weights = {i: total_samples / (len(unique_classes_client) * count) if count > 0 else 0
                         for i, count in enumerate(class_counts)}

        print(f"Client {client_idx + 1}: {len(client_X)} samples, {len(unique_classes_client)} unique students")

        client_data.append({
            'X': client_X,
            'y': client_y,
            'class_weights': class_weights,
            'num_classes': len(label_encoder.classes_)
        })

    return client_data, X_test_scaled, y_test, scaler, label_encoder, id_protector


def create_original_deep_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=(input_shape,))

    # First block
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(256, kernel_regularizer=l2(L2_REG))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    # Intermediate layer
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2(L2_REG))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    # Residual block
    skip = x
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2(L2_REG))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    x = tf.keras.layers.Dense(128, kernel_regularizer=l2(L2_REG))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.add([x, skip])
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_client_model(client_data, global_model_weights, input_shape, n_folds=NUM_FOLDS, epochs=NUM_EPOCHS):
    """Train model on client data using cross-validation"""

    print(f"\nTraining client model with {len(client_data['X'])} samples...")
    X_train = client_data['X']
    y_train = client_data['y']
    class_weights = client_data['class_weights']
    num_classes = client_data['num_classes']

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_fold_models = []
    val_accuracies = []
    all_histories = []
    training_times = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nTraining fold {fold + 1}/{n_folds}")

        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train[val_idx]

        model = create_original_deep_model(input_shape, num_classes)

        if global_model_weights is not None:
            try:
                model.set_weights(global_model_weights)
            except ValueError as e:
                print(f"Cannot apply global weights: {e}")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        start_time = time.time()
        history = model.fit(
            X_train_fold.values, y_train_fold,
            epochs=epochs,
            batch_size=64,
            validation_data=(X_val_fold.values, y_val_fold),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        training_time = time.time() - start_time
        training_times.append(training_time)
        all_histories.append(history.history)

        val_loss, val_accuracy = model.evaluate(X_val_fold.values, y_val_fold, verbose=0)
        val_accuracies.append(val_accuracy)
        all_fold_models.append(model)

        print(f"Fold {fold + 1} - Validation Accuracy: {val_accuracy:.4f}")

    best_model_idx = np.argmax(val_accuracies)
    best_model = all_fold_models[best_model_idx]

    print(f"Best fold model accuracy: {val_accuracies[best_model_idx]:.4f}")

    return best_model, val_accuracies[best_model_idx], all_histories, training_times


def aggregate_models(models, weights=None):
    if weights is None:
        weights = [1.0] * len(models)

    weights = np.array(weights) / np.sum(weights)
    global_weights = [np.zeros_like(w) for w in models[0].get_weights()]

    for model_idx, model in enumerate(models):
        model_weights = model.get_weights()
        for i, layer_weights in enumerate(model_weights):
            global_weights[i] += layer_weights * weights[model_idx]

    return global_weights


def federated_learning(client_data, X_test, y_test, input_shape, n_rounds=NUM_ROUNDS, n_folds=NUM_FOLDS,
                       epochs=NUM_EPOCHS):
    """Perform federated learning"""

    num_clients = len(client_data)
    num_classes = max([client['num_classes'] for client in client_data])

    print(f"\nStarting federated learning with {num_clients} clients")

    global_model = create_original_deep_model(input_shape, num_classes)
    global_weights = global_model.get_weights()

    round_accuracies = []
    all_client_histories = []
    round_training_times = []
    round_testing_times = []

    for round_idx in range(n_rounds):
        print(f"\n===== FEDERATED ROUND {round_idx + 1}/{n_rounds} =====")
        round_start_time = time.time()

        client_models = []
        client_accuracies = []
        client_sample_sizes = []
        round_client_histories = []

        for client_idx, client in enumerate(client_data):
            print(f"\n----- Client {client_idx + 1} Training -----")
            client_model, client_accuracy, client_histories, client_training_times = train_client_model(
                client, global_weights, input_shape, n_folds=n_folds, epochs=epochs
            )
            client_models.append(client_model)
            client_accuracies.append(client_accuracy)
            client_sample_sizes.append(len(client['X']))
            round_client_histories.append(client_histories)

        global_weights = aggregate_models(client_models, weights=client_sample_sizes)
        global_model.set_weights(global_weights)

        test_start_time = time.time()
        test_loss, test_accuracy = global_model.evaluate(X_test.values, y_test, verbose=0)
        test_time = time.time() - test_start_time

        round_total_time = time.time() - round_start_time

        round_accuracies.append(test_accuracy)
        all_client_histories.append(round_client_histories)
        round_training_times.append(round_total_time - test_time)
        round_testing_times.append(test_time)

        print(f"\nRound {round_idx + 1} complete - Global test accuracy: {test_accuracy:.4f}")

    global_model.save('models/federated_global_model.h5')
    return global_model, round_accuracies, all_client_histories, round_training_times, round_testing_times


def plot_training_validation_curves(all_client_histories, round_num, client_num):

    if not all_client_histories or round_num >= len(all_client_histories):
        return

    if round_num < 0:
        round_num = len(all_client_histories) + round_num

    round_histories = all_client_histories[round_num]
    if client_num >= len(round_histories):
        return

    client_histories = round_histories[client_num]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training/Validation Curves - Round {round_num + 1}, Client {client_num + 1}', fontsize=16)

    all_train_acc = []
    all_val_acc = []
    all_train_loss = []
    all_val_loss = []

    for fold_idx, history in enumerate(client_histories):
        if fold_idx < 3:
            epochs = range(1, len(history['accuracy']) + 1)

            axes[0, 0].plot(epochs, history['accuracy'], f'C{fold_idx}-', alpha=0.7, label=f'Fold {fold_idx + 1} Train')
            axes[0, 0].plot(epochs, history['val_accuracy'], f'C{fold_idx}--', alpha=0.7,
                            label=f'Fold {fold_idx + 1} Val')

            axes[0, 1].plot(epochs, history['loss'], f'C{fold_idx}-', alpha=0.7, label=f'Fold {fold_idx + 1} Train')
            axes[0, 1].plot(epochs, history['val_loss'], f'C{fold_idx}--', alpha=0.7, label=f'Fold {fold_idx + 1} Val')

        all_train_acc.append(history['accuracy'])
        all_val_acc.append(history['val_accuracy'])
        all_train_loss.append(history['loss'])
        all_val_loss.append(history['val_loss'])

    axes[0, 0].set_title('Training/Validation Accuracy by Fold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Training/Validation Loss by Fold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if all_train_acc:
        min_epochs = min(len(acc) for acc in all_train_acc)

        train_acc_avg = np.mean([acc[:min_epochs] for acc in all_train_acc], axis=0)
        val_acc_avg = np.mean([acc[:min_epochs] for acc in all_val_acc], axis=0)
        train_loss_avg = np.mean([loss[:min_epochs] for loss in all_train_loss], axis=0)
        val_loss_avg = np.mean([loss[:min_epochs] for loss in all_val_loss], axis=0)

        epochs_avg = range(1, min_epochs + 1)

        axes[1, 0].plot(epochs_avg, train_acc_avg, 'b-', linewidth=2, label='Average Train Accuracy')
        axes[1, 0].plot(epochs_avg, val_acc_avg, 'r-', linewidth=2, label='Average Val Accuracy')
        axes[1, 0].set_title('Average Training/Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs_avg, train_loss_avg, 'b-', linewidth=2, label='Average Train Loss')
        axes[1, 1].plot(epochs_avg, val_loss_avg, 'r-', linewidth=2, label='Average Val Loss')
        axes[1, 1].set_title('Average Training/Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/training_curves_round_{round_num + 1}_client_{client_num + 1}.png', dpi=300,
                bbox_inches='tight')
    plt.close()


def plot_federated_rounds_comparison(all_client_histories, round_accuracies):

    if not all_client_histories:
        return

    num_rounds = len(all_client_histories)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Federated Learning Performance Summary', fontsize=16)

    rounds = list(range(1, num_rounds + 1))
    avg_final_train_acc = []
    avg_final_val_acc = []
    avg_final_train_loss = []
    avg_final_val_loss = []

    for round_idx in range(num_rounds):
        round_train_accs = []
        round_val_accs = []
        round_train_losses = []
        round_val_losses = []

        for client_idx in range(len(all_client_histories[round_idx])):
            client_histories = all_client_histories[round_idx][client_idx]

            for history in client_histories:
                if history['accuracy']:
                    round_train_accs.append(history['accuracy'][-1])
                    round_val_accs.append(history['val_accuracy'][-1])
                    round_train_losses.append(history['loss'][-1])
                    round_val_losses.append(history['val_loss'][-1])

        if round_train_accs:
            avg_final_train_acc.append(np.mean(round_train_accs))
            avg_final_val_acc.append(np.mean(round_val_accs))
            avg_final_train_loss.append(np.mean(round_train_losses))
            avg_final_val_loss.append(np.mean(round_val_losses))
        else:
            avg_final_train_acc.append(0)
            avg_final_val_acc.append(0)
            avg_final_train_loss.append(0)
            avg_final_val_loss.append(0)

    axes[0, 0].plot(rounds, avg_final_train_acc, 'b-o', linewidth=2, label='Training Accuracy', markersize=8)
    axes[0, 0].plot(rounds, avg_final_val_acc, 'r-o', linewidth=2, label='Validation Accuracy', markersize=8)
    axes[0, 0].set_title('Average Final Accuracy per Round')
    axes[0, 0].set_xlabel('Federated Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(rounds, avg_final_train_loss, 'b-o', linewidth=2, label='Training Loss', markersize=8)
    axes[0, 1].plot(rounds, avg_final_val_loss, 'r-o', linewidth=2, label='Validation Loss', markersize=8)
    axes[0, 1].set_title('Average Final Loss per Round')
    axes[0, 1].set_xlabel('Federated Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(rounds, round_accuracies, 'g-o', linewidth=3, label='Global Test Accuracy', markersize=10)
    axes[1, 0].axhline(y=max(round_accuracies), color='g', linestyle='--', alpha=0.7,
                       label=f'Best: {max(round_accuracies):.4f}')
    axes[1, 0].set_title('Global Model Test Accuracy')
    axes[1, 0].set_xlabel('Federated Round')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if len(round_accuracies) > 1:
        improvements = [round_accuracies[i] - round_accuracies[i - 1] for i in range(1, len(round_accuracies))]
        axes[1, 1].bar(rounds[1:], improvements, alpha=0.7, color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Accuracy Improvement per Round')
        axes[1, 1].set_xlabel('Federated Round')
        axes[1, 1].set_ylabel('Accuracy Gain')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/federated_summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(model, X_test, y_test, label_encoder, num_classes):

    y_pred_proba = model.predict(X_test.values)
    y_test_bin = label_binarize(y_test, classes=range(num_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        if np.sum(y_test_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(12, 8))

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=2)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green',
                    'purple', 'brown', 'pink', 'gray', 'olive'])

    classes_to_plot = min(10, len(roc_auc))
    for i, color in zip(range(classes_to_plot), colors):
        if i in roc_auc:
            class_name = label_encoder.classes_[i] if hasattr(label_encoder, 'classes_') else f'Class {i}'
            plt.plot(fpr[i], tpr[i], color=color, linewidth=1,
                     label=f'{class_name} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Federated Global Model')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_timing_analysis(round_training_times, round_testing_times):
    """Plot training and testing time analysis"""

    rounds = range(1, len(round_training_times) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Testing Time Analysis', fontsize=16)

    axes[0, 0].plot(rounds, round_training_times, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_title('Training Time per Round')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(rounds, round_testing_times, 'r-o', linewidth=2, markersize=8)
    axes[0, 1].set_title('Testing Time per Round')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar([r - 0.2 for r in rounds], round_training_times, 0.4, label='Training', alpha=0.7)
    axes[1, 0].bar([r + 0.2 for r in rounds], round_testing_times, 0.4, label='Testing', alpha=0.7)
    axes[1, 0].set_title('Training vs Testing Time per Round')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    cumulative_training = np.cumsum(round_training_times)
    cumulative_testing = np.cumsum(round_testing_times)

    axes[1, 1].plot(rounds, cumulative_training, 'b-o', label='Cumulative Training', linewidth=2)
    axes[1, 1].plot(rounds, cumulative_testing, 'r-o', label='Cumulative Testing', linewidth=2)
    axes[1, 1].set_title('Cumulative Time Analysis')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Cumulative Time (seconds)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/timing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_federated_results(round_accuracies, client_data, test_acc, model, X_test, y_test, label_encoder,
                                all_client_histories, round_training_times, round_testing_times):
    """Here i create all visualizations"""

    # 1. Federated learning progression
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(round_accuracies) + 1), round_accuracies, marker='o', linewidth=2)
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Final Accuracy: {test_acc:.4f}')
    plt.title('Federated Learning Accuracy Progression')
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/federated_learning_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Client data distribution
    plt.figure(figsize=(12, 6))
    for i, client in enumerate(client_data):
        unique_counts = np.bincount(client['y'])
        plt.bar(
            x=np.arange(len(unique_counts)) + (i * 0.3),
            height=unique_counts,
            width=0.3,
            alpha=0.7,
            label=f'Client {i + 1}'
        )
    plt.title('Client Data Distribution by Student ID')
    plt.xlabel('Student ID (encoded)')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/client_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Confusion matrix (percentage)
    y_pred = model.predict(X_test.values)
    y_pred_classes = np.argmax(y_pred, axis=1)

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred_classes)

    # Convert to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    if len(cm) > 10:
        top_classes = np.argsort(np.diag(cm))[-10:]
        cm_percentage = cm_percentage[top_classes, :][:, top_classes]
        class_names = label_encoder.classes_[top_classes]
    else:
        class_names = label_encoder.classes_

    sns.heatmap(cm_percentage,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})

    plt.title('Confusion Matrix - Percentage (Federated Global Model)')
    plt.ylabel('True Student ID')
    plt.xlabel('Predicted Student ID')

    plt.figtext(0.02, 0.02, 'Values shown as percentages of true class',
                fontsize=10, style='italic', alpha=0.7)

    plt.tight_layout()
    plt.savefig('results/federated_confusion_matrix_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Training/Validation curves for ALL clients
    if all_client_histories:
        print("\nGenerating training/validation curves for all clients...")

        # Generate curves for first round - all clients
        for client_idx in range(len(client_data)):
            plot_training_validation_curves(all_client_histories, 0, client_idx)
            print(f"✓ Generated: First round curves for Client {client_idx + 1}")

        for client_idx in range(len(client_data)):
            plot_training_validation_curves(all_client_histories, -1, client_idx)
            print(f"✓ Generated: Final round curves for Client {client_idx + 1}")

        plot_federated_rounds_comparison(all_client_histories, round_accuracies)
        print("✓ Generated: Federated rounds comparison summary")

    num_classes = model.output_shape[1]
    plot_roc_curves(model, X_test, y_test, label_encoder, num_classes)

    # 6. Timing analysis
    plot_timing_analysis(round_training_times, round_testing_times)


def predict_student_id(features, model_path='models/federated_global_model.h5',
                       scaler_path='models/student_id_feature_scaler.pkl',
                       encoder_path='models/student_id_dummy_encoder.pkl'):
    """Predict student ID for new data"""

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    if not isinstance(features, pd.DataFrame):
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_
        else:
            raise ValueError("Features must be a DataFrame with named columns")
        features = pd.DataFrame([features], columns=feature_names)

    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    dummy_student_id = label_encoder.inverse_transform([predicted_class])[0]

    user_input = input("\nDo you want to see the original student ID? (y/n): ")
    if user_input.lower() == "y":
        password = input("Enter admin password: ")
        if password == ADMIN_PASSWORD:
            id_protector = SimpleIDProtector.load_mapping()
            if id_protector:
                original_id = id_protector.dummy_to_original_id(dummy_student_id)
                if original_id:
                    print(f"Original student ID: {original_id}")
                else:
                    print(f"No mapping found for dummy ID: {dummy_student_id}")
            else:
                print("Error: Could not load ID mapping")
        else:
            print("Access denied: Incorrect password")

    return {
        'predicted_user': str(dummy_student_id),
        'confidence': float(confidence),
    }


def main():
    data_path = "privacyofstudents.csv"
    client_data, X_test, y_test, scaler, label_encoder, id_protector = dataprepration(
        data_path, test_size=0.2, num_clients=NUM_CLIENTS, random_state=42
    )

    input_shape = X_test.shape[1]
    global_model, round_accuracies, all_client_histories, round_training_times, round_testing_times = federated_learning(
        client_data, X_test, y_test, input_shape, n_rounds=NUM_ROUNDS, n_folds=NUM_FOLDS, epochs=NUM_EPOCHS
    )

    test_loss, test_accuracy = global_model.evaluate(X_test.values, y_test, verbose=0)
    print(f"\nFinal federated model evaluation:")
    print(f"Test accuracy: {test_accuracy:.4f}")

    model_metadata = {
        'feature_names': X_test.columns.tolist(),
        'num_features': X_test.shape[1],
        'num_classes': global_model.output_shape[1],
        'accuracy': float(test_accuracy),
        'federated_rounds': NUM_ROUNDS,
        'clients': NUM_CLIENTS,
        'cross_validation_folds': NUM_FOLDS,
        'epochs_per_round': NUM_EPOCHS,
        'total_training_time': sum(round_training_times),
        'total_testing_time': sum(round_testing_times)
    }
    joblib.dump(model_metadata, 'models/federated_model_metadata.pkl')
    visualize_federated_results(
        round_accuracies, client_data, test_accuracy, global_model,
        X_test, y_test, label_encoder, all_client_histories,
        round_training_times, round_testing_times
    )

    print("\n How TO USE:")
    print("   • Run '.py file' to test predictions")
    print("   • Admin password required for original student IDs (Currently harcoded)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()