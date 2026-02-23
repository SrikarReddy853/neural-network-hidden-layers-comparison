import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# 1. Generate nonlinear dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)


# 2. Split data (30% for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# 3. Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4. Function to build model with n hidden layers
def build_model(n_hidden_layers):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(2,)))
    for _ in range(n_hidden_layers - 1):
        model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam()
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 5. Train models with 1, 2, and 3 hidden layers
models = []
histories = []
n_layers_list = [1, 2, 3]
for n_layers in n_layers_list:
    print(f"\nTraining model with {n_layers} hidden layer(s)...")
    model = build_model(n_layers)
    history = model.fit(
        X_train_scaled, y_train,
        epochs=20,
        batch_size=32,
        verbose=1,
        validation_split=0.2
    )
    models.append(model)
    histories.append(history)

# 6. Plot loss and validation accuracy curves
epochs = range(1, 21)
plt.figure(figsize=(14, 6))

# Plot 1: Train and Validation Loss for 1, 2, and 3 hidden layers
plt.subplot(1, 2, 1)
for i, n_layers in enumerate(n_layers_list):
    plt.plot(epochs, histories[i].history['loss'],
             label=f'Train Loss - {n_layers} layer(s)')
    plt.plot(epochs, histories[i].history['val_loss'], linestyle='--',
             label=f'Val Loss - {n_layers} layer(s)')
plt.title('Loss vs Epochs (All Models)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot 2: Validation Accuracy vs Epoch for 1, 2, and 3 hidden layers
plt.subplot(1, 2, 2)
for i, n_layers in enumerate(n_layers_list):
    plt.plot(epochs, histories[i].history['val_accuracy'],
             label=f'Val Acc - {n_layers} layer(s)')
plt.title('Validation Accuracy vs Epoch (All Models)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 7. Evaluate on test set
for i, model in enumerate(models, start=1):
    loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy of model with {i} hidden layer(s): {acc:.3f}")
