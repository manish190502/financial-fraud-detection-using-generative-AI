import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam

# Load your dataset
data = pd.read_csv('creditcard.csv')

# Preprocessing
X = data.drop('Class', axis=1).values  # Features
y = data['Class'].values.reshape(-1, 1)  # Target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Generator model
gen_input_dim = X_train_scaled.shape[1]
gen_input = Input(shape=(gen_input_dim,))
gen = Dense(128, activation='relu')(gen_input)
gen = Dense(64, activation='relu')(gen)
gen_output = Dense(gen_input_dim, activation='tanh')(gen)
generator = Model(gen_input, gen_output)

# Discriminator model
disc_input_dim = gen_input_dim + 1  # Features + Class label
disc_input = Input(shape=(disc_input_dim,))
disc = Dense(128, activation='relu')(disc_input)
disc = Dense(64, activation='relu')(disc)
disc_output = Dense(1, activation='sigmoid')(disc)
discriminator = Model(disc_input, disc_output)
discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Combined model (CGAN)
cgan_input = Input(shape=(gen_input_dim,))
label_input = Input(shape=(1,))
concatenated_input = Concatenate()([cgan_input, label_input])
cgan_output = discriminator(concatenated_input)
cgan = Model(inputs=[cgan_input, label_input], outputs=cgan_output)
cgan.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Train CGAN
batch_size = 128
epochs = 10000
for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, X_train_scaled.shape[0], batch_size)
    real_transactions = X_train_scaled[idx]
    real_labels = y_train[idx]
    fake_labels = np.random.randint(0, 2, (batch_size, 1))
    fake_transactions = generator.predict(np.random.normal(0, 1, (batch_size, gen_input_dim)))
    disc_loss_real = discriminator.train_on_batch(np.concatenate([real_transactions, real_labels], axis=1), np.ones((batch_size, 1)))
    disc_loss_fake = discriminator.train_on_batch(np.concatenate([fake_transactions, fake_labels], axis=1), np.zeros((batch_size, 1)))
    disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, gen_input_dim))
    valid_y = np.ones((batch_size, 1))
    gen_loss = cgan.train_on_batch([noise, fake_labels], valid_y)

    # Print progress
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Disc_loss: {disc_loss[0]}, Gen_loss: {gen_loss}')

# Generate synthetic data
num_synthetic_samples = 1000
synthetic_noise = np.random.normal(0, 1, (num_synthetic_samples, gen_input_dim))
synthetic_labels = np.random.randint(0, 2, (num_synthetic_samples, 1))
synthetic_data = generator.predict(synthetic_noise)

# Save synthetic data to a CSV file
synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns[:-1])  # Assuming last column is the target 'Class'
synthetic_df.to_csv('gan.csv', index=False)