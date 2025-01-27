import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
import joblib

class DroughtPredictionSystem:
    def __init__(self):
        self.coordinates_scaler = StandardScaler()
        self.temporal_scaler = StandardScaler()
        self.lstm_model = None
        
    def prepare_data(self, df):
        coordinate_features = ['LAT', 'LON', 'BOTTOM_LEFT_LAT', 'BOTTOM_LEFT_LON', 
                             'UPPER_RIGHT_LAT', 'UPPER_RIGHT_LON']
        normalized_static_features = ['DRAIN', 'CFRAG', 'SDTO', 'STPC', 'CLPC', 'PSCL', 
                                    'BULK', 'TAWC', 'CECS', 'BSAT', 'CECC', 'PHAQ', 
                                    'TCEQ', 'GYPS', 'ELCO', 'TOTC', 'TOTN', 'ECEC', 
                                    'ALSA', 'ESP']
        temporal_features = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        X_coordinates = df[coordinate_features].values
        X_coordinates_scaled = self.coordinates_scaler.fit_transform(X_coordinates)
        X_normalized = df[normalized_static_features].values
        X_static = np.hstack([X_coordinates_scaled, X_normalized])
        X_temporal = df[temporal_features].values
        X_temporal_scaled = self.temporal_scaler.fit_transform(X_temporal)
        X_temporal_reshaped = X_temporal_scaled.reshape(-1, 12, 1)
        y = df['ANN'].values
        
        return X_static, X_temporal_reshaped, y
    
    def build_hybrid_model(self, static_input_dim):
        static_input = Input(shape=(static_input_dim,))
        temporal_input = Input(shape=(12, 1))
        lstm_out = LSTM(64, return_sequences=False)(temporal_input)
        combined = Concatenate()([static_input, lstm_out])
        dense1 = Dense(128, activation='relu')(combined)
        dense2 = Dense(64, activation='relu')(dense1)
        output = Dense(1)(dense2)
        self.lstm_model = Model(inputs=[static_input, temporal_input], outputs=output)
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')  # Changed from 'mse' to 'mean_squared_error'
        
    def train(self, df, validation_split=0.2):
        X_static, X_temporal, y = self.prepare_data(df)
        X_static_train, X_static_val, X_temporal_train, X_temporal_val, y_train, y_val = \
            train_test_split(X_static, X_temporal, y, test_size=validation_split, random_state=42)
        self.build_hybrid_model(X_static.shape[1])
        history = self.lstm_model.fit(
            [X_static_train, X_temporal_train],
            y_train,
            validation_data=([X_static_val, X_temporal_val], y_val),
            epochs=50,
            batch_size=32
        )
        return history
    
    def predict(self, df):
        coordinate_features = ['LAT', 'LON', 'BOTTOM_LEFT_LAT', 'BOTTOM_LEFT_LON', 
                             'UPPER_RIGHT_LAT', 'UPPER_RIGHT_LON']
        normalized_static_features = ['DRAIN', 'CFRAG', 'SDTO', 'STPC', 'CLPC', 'PSCL', 
                                    'BULK', 'TAWC', 'CECS', 'BSAT', 'CECC', 'PHAQ', 
                                    'TCEQ', 'GYPS', 'ELCO', 'TOTC', 'TOTN', 'ECEC', 
                                    'ALSA', 'ESP']
        temporal_features = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        X_coordinates = df[coordinate_features].values
        X_coordinates_scaled = self.coordinates_scaler.transform(X_coordinates)
        X_normalized = df[normalized_static_features].values
        X_static = np.hstack([X_coordinates_scaled, X_normalized])
        X_temporal = df[temporal_features].values
        X_temporal_scaled = self.temporal_scaler.transform(X_temporal)
        X_temporal_reshaped = X_temporal_scaled.reshape(-1, 12, 1)
        prediction = self.lstm_model.predict([X_static, X_temporal_reshaped])
        return prediction
    
    def save_models(self, path_prefix):
        save_model(self.lstm_model, f"{path_prefix}_lstm.h5")
        joblib.dump(self.coordinates_scaler, f"{path_prefix}_coordinates_scaler.joblib")
        joblib.dump(self.temporal_scaler, f"{path_prefix}_temporal_scaler.joblib")

