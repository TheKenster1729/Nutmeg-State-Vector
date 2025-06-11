import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow_probability as tfp
import pickle
import os

tfd = tfp.distributions

class MDN(keras.Model):
    """Mixed Density Network for predicting locations from street names."""
    
    def __init__(self, vocab_size, num_mixtures=5):
        """
        Initialize MDN model.
        
        Args:
            vocab_size: Size of character vocabulary
            num_mixtures: Number of Gaussian components in the mixture
        """
        super(MDN, self).__init__()
        self.num_mixtures = num_mixtures
        self.embedding = layers.Embedding(vocab_size, 64)
        self.lstm = layers.LSTM(128)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        
        # Output layers for MDN parameters
        self.pi_layer = layers.Dense(num_mixtures, activation='softmax')  # mixture weights
        self.mu_layer = layers.Dense(num_mixtures * 2)  # means (2D: lon, lat)
        self.sigma_layer = layers.Dense(num_mixtures * 2, activation='softplus')  # standard deviations
        self.corr_layer = layers.Dense(num_mixtures, activation='tanh')  # correlations
        
    def call(self, inputs):
        """Forward pass through the model."""
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        pi = self.pi_layer(x)
        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)
        corr = self.corr_layer(x)
        
        return pi, mu, sigma, corr

class TrainModel:
    """Class for training and using a street name location prediction model."""
    
    def __init__(self, db_connection_string=None):
        """
        Initialize the TrainModel class.
        
        Args:
            db_connection_string: Connection string for PostgreSQL database
        """
        self.db_connection_string = db_connection_string or 'postgresql://postgres:5342@localhost:5432/ctroads'
        self.max_text_len = 20
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = None
        self.coord_scaler = StandardScaler()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        # Set TensorFlow to use float32 as default
        tf.keras.backend.set_floatx('float32')
        
        # Training parameters
        self.batch_size = 64
        self.num_epochs = 20
        self.num_mixtures = 5
        
        # Data containers
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_scaled = None
        self.y_test_scaled = None
        self.train_street_names = None
        self.test_street_names = None
        
    def load_data(self):
        """Load data from PostgreSQL database."""
        engine = create_engine(self.db_connection_string)
        
        # Fetch data from database
        query = """
        SELECT "FULLNAME", "NORMSTREETNAME", "lon", "lat" 
        FROM roads
        """
        self.df = pd.read_sql(query, engine)
        
        self.df = self.df[self.df['NORMSTREETNAME'] != '']
        print(f"Dataset shape: {self.df.shape}")

        # Apply preprocessing
        self.df['processed_name'] = self.df['NORMSTREETNAME'].apply(self.preprocess_text)
        
        # Create vocabulary
        self._create_vocabulary()
        
    def _create_vocabulary(self):
        """Create character-level vocabulary from street names."""
        unique_chars = set()
        for name in self.df['processed_name']:
            unique_chars.update(name)
        
        self.char_to_idx = {c: i+1 for i, c in enumerate(sorted(unique_chars))}  # 0 is reserved for padding
        self.idx_to_char = {i+1: c for i, c in enumerate(sorted(unique_chars))}
        self.vocab_size = len(self.char_to_idx) + 1  # +1 for padding
        
    def preprocess_text(self, text):
        """
        Preprocess street name text.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        # You might want to add more preprocessing steps here
        return text
        
    def text_to_sequence(self, text, max_len=None):
        """
        Convert text to sequence of indices.
        
        Args:
            text: Text to convert
            max_len: Maximum sequence length
            
        Returns:
            Sequence of indices
        """
        max_len = max_len or self.max_text_len
        sequence = [self.char_to_idx.get(c, 0) for c in text]
        
        # Pad or truncate to max_len
        if len(sequence) < max_len:
            sequence = sequence + [0] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
            
        return sequence
    
    def prepare_data(self):
        """Prepare data for training by creating features and splitting into train/test sets."""
        # Create feature and target arrays
        X_text = np.array([self.text_to_sequence(name) for name in self.df['processed_name']])
        y = self.df[['lon', 'lat']].values.astype(np.float32)  # Convert to float32
        
        # Create a dataset with indices and keep track of original street names
        indices = np.arange(len(X_text))
        street_names = self.df['NORMSTREETNAME'].values
        
        # Split into train/test while keeping track of indices
        X_train_idx, X_test_idx, y_train, y_test, train_indices, test_indices = train_test_split(
            X_text, y, indices, test_size=0.2, random_state=42
        )
        
        # Get the corresponding street names for train and test sets
        self.train_street_names = street_names[train_indices]
        self.test_street_names = street_names[test_indices]
        
        # Export street names to CSV
        train_streets_df = pd.DataFrame({'street_name': self.train_street_names, 'lat': y_train[:, 1], 'lon': y_train[:, 0]})
        test_streets_df = pd.DataFrame({'street_name': self.test_street_names, 'lat': y_test[:, 1], 'lon': y_test[:, 0]})
        
        train_streets_df.to_csv('training_set_streets.csv', index=False)
        test_streets_df.to_csv('test_set_streets.csv', index=False)
        
        print(f"Exported {len(self.train_street_names)} training streets to training_set_streets.csv")
        print(f"Exported {len(self.test_street_names)} test streets to test_set_streets.csv")
        
        # Normalize coordinates and ensure float32
        self.y_train_scaled = self.coord_scaler.fit_transform(y_train).astype(np.float32)
        self.y_test_scaled = self.coord_scaler.transform(y_test).astype(np.float32)
        
        # Store data
        self.X_train = X_train_idx
        self.X_test = X_test_idx
        self.y_train = y_train
        self.y_test = y_test

        print(self.X_train)
        print(self.y_train)
        
        # print(f"Training set shape: {self.X_train.shape}, {self.y_train.shape}")
        # print(f"Test set shape: {self.X_test.shape}, {self.y_test.shape}")
        # print(f"Data types - X_train: {self.X_train.dtype}, y_train_scaled: {self.y_train_scaled.dtype}")
        
    def create_model(self):
        """Create the MDN model."""
        self.model = MDN(self.vocab_size, num_mixtures=self.num_mixtures)
        
        # Initialize model by calling it once
        dummy_input = tf.zeros((1, self.max_text_len), dtype=tf.int32)
        _ = self.model(dummy_input)
        
    def mdn_loss(self, y_true, mdn_params):
        """
        Custom loss function for MDN.
        
        Args:
            y_true: True coordinates
            mdn_params: Tuple of (pi, mu, sigma, corr)
            
        Returns:
            Negative log likelihood loss
        """
        pi, mu, sigma, corr = mdn_params
        
        # Ensure consistent data types (convert all to float32)
        y_true = tf.cast(y_true, tf.float32)
        pi = tf.cast(pi, tf.float32)
        mu = tf.cast(mu, tf.float32)
        sigma = tf.cast(sigma, tf.float32)
        corr = tf.cast(corr, tf.float32)
        
        # Reshape parameters
        batch_size = tf.shape(y_true)[0]
        y_true = tf.reshape(y_true, [batch_size, 2])
        
        mu = tf.reshape(mu, [batch_size, -1, 2])
        sigma = tf.reshape(sigma, [batch_size, -1, 2])
        corr = tf.reshape(corr, [batch_size, -1])
        
        # Create mixture of bivariate normal distributions
        mix = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pi),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=mu,
                scale_tril=tfp.math.fill_triangular(
                    tf.concat(
                        [
                            tf.math.log(sigma[..., 0:1]),
                            corr[..., tf.newaxis],
                            tf.math.log(sigma[..., 1:2])
                        ],
                        axis=-1
                    )
                )
            )
        )
        
        # Calculate negative log likelihood
        loss = -tf.reduce_mean(mix.log_prob(y_true))
        return loss
    
    @tf.function
    def train_step(self, X, y):
        """
        Single training step with gradient update.
        
        Args:
            X: Input batch
            y: Target batch
            
        Returns:
            Loss value
        """
        with tf.GradientTape() as tape:
            pi, mu, sigma, corr = self.model(X)
            loss = self.mdn_loss(y, (pi, mu, sigma, corr))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def train_model(self):
        """Train the MDN model."""
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_train, self.y_train_scaled)
        ).batch(self.batch_size)
        
        # Train the model
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            steps = 0
            
            for X_batch, y_batch in train_dataset:
                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
                steps += 1
            
            avg_loss = epoch_loss / steps
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
            
        # Evaluate on test set
        self.evaluate_model()
        
    def evaluate_model(self):
        """Evaluate the model on test data."""
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_test, self.y_test_scaled)
        ).batch(self.batch_size)
        
        test_loss = 0
        steps = 0
        
        for X_batch, y_batch in test_dataset:
            pi, mu, sigma, corr = self.model(X_batch)
            loss = self.mdn_loss(y_batch, (pi, mu, sigma, corr))
            test_loss += loss
            steps += 1
        
        avg_test_loss = test_loss / steps
        print(f"Test Loss: {avg_test_loss:.4f}")
    
    def save_model(self, model_dir='model_files'):
        """
        Save model weights, vocabulary, and scaler.
        
        Args:
            model_dir: Directory to save model files
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model weights
        weights_path = os.path.join(model_dir, 'street_name_mdn_model.weights.h5')
        self.model.save_weights(weights_path)
        
        # Save vocabulary
        vocab_path = os.path.join(model_dir, 'street_name_vocab.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'max_text_len': self.max_text_len
            }, f)
        
        # Save coordinate scaler
        scaler_path = os.path.join(model_dir, 'coord_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.coord_scaler, f)
            
        print(f"Model saved to {model_dir}/")
    
    def load_model(self, model_dir='model_files'):
        """
        Load model weights, vocabulary, and scaler.
        
        Args:
            model_dir: Directory with model files
        """
        # Load vocabulary
        vocab_path = os.path.join(model_dir, 'street_name_vocab.pkl')
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = vocab_data['idx_to_char']
        self.max_text_len = vocab_data['max_text_len']
        self.vocab_size = len(self.char_to_idx) + 1
        
        # Load coordinate scaler
        scaler_path = os.path.join(model_dir, 'coord_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.coord_scaler = pickle.load(f)
        
        # Create and load model
        self.create_model()
        weights_path = os.path.join(model_dir, 'street_name_mdn_model.weights.h5')
        self.model.load_weights(weights_path)
        
        print(f"Model loaded from {model_dir}/")
    
    def predict_location(self, street_name):
        """
        Predict location for a street name.
        
        Args:
            street_name: Name of the street
            
        Returns:
            Predicted longitude and latitude as [lon, lat]
        """
        # Preprocess and convert to sequence
        processed_name = self.preprocess_text(street_name)
        sequence = np.array([self.text_to_sequence(processed_name)], dtype=np.int32)
        
        # Get prediction
        pi, mu, sigma, corr = self.model(sequence)
        
        # Cast to float32 for consistency
        pi = tf.cast(pi, tf.float32)
        mu = tf.cast(mu, tf.float32)
        
        # Reshape parameters
        mu = tf.reshape(mu, [1, -1, 2])
        
        # Get most likely component
        most_likely_idx = tf.argmax(pi, axis=1)[0]
        most_likely_mu = mu[0, most_likely_idx]
        
        # Convert back to original scale
        predicted_loc = self.coord_scaler.inverse_transform(most_likely_mu.numpy().reshape(1, -1))[0]
        
        return predicted_loc
    
    def plot_prediction(self, street_name, predicted_loc=None, show_all_streets=True):
        """
        Plot prediction for a street name on a map.
        
        Args:
            street_name: Name of the street
            predicted_loc: Predicted location (if None, will be predicted)
            show_all_streets: Whether to show all streets as background
        """
        if predicted_loc is None:
            predicted_loc = self.predict_location(street_name)
            
        # Get actual locations for this street name
        print(self.df['NORMSTREETNAME'])
        actual_locs = self.df[self.df['NORMSTREETNAME'].str.lower() == street_name.lower()][['lon', 'lat']].values
        
        plt.figure(figsize=(10, 8))
        
        # Plot all streets in Connecticut as background (small dots)
        if show_all_streets:
            plt.scatter(self.df['lon'], self.df['lat'], s=1, color='gray', alpha=0.1)
        
        # Plot actual locations for this street
        if len(actual_locs) > 0:
            plt.scatter(actual_locs[:, 0], actual_locs[:, 1], 
                      s=30, color='blue', label=f'Actual {street_name} locations')
        
        # Plot predicted location
        plt.scatter(predicted_loc[0], predicted_loc[1], 
                  s=100, color='red', marker='*', label=f'Predicted {street_name}')
        
        plt.title(f'Prediction for "{street_name}" in Connecticut')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        fig_path = f"{street_name.replace(' ', '_')}_prediction.png"
        plt.savefig(fig_path)
        print(f"Saved prediction plot to {fig_path}")
        
        plt.show()
        
    def run_pipeline(self):
        """Run the full machine learning pipeline."""
        self.load_data()
        self.prepare_data()
        self.create_model()
        self.train_model()
        self.save_model()
        
        # Example prediction
        test_street = "Hine Street"
        predicted_loc = self.predict_location(test_street)
        print(f"\nMost likely location for '{test_street}':")
        print(f"Longitude: {predicted_loc[0]:.6f}, Latitude: {predicted_loc[1]:.6f}")
        
        # Plot prediction
        self.plot_prediction(test_street, predicted_loc)

# Example usage
if __name__ == "__main__":
    # Create and train model
    trainer = TrainModel()
    trainer.run_pipeline()
