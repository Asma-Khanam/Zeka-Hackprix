# arduino_bioamp_bci.py - Arduino-based BioAmp EEG BCI System
import numpy as np
import joblib
import json
import time
import serial
from collections import deque
import threading
from scipy import signal
import matplotlib.pyplot as plt

class ArduinoBioAmpBCI:
    def __init__(self, arduino_port='COM3', sampling_rate=250):
        print("🧠 === Arduino BioAmp BCI System ===")
        
        # Load trained models
        self.load_models()
        
        # Arduino connection
        self.arduino_port = arduino_port
        self.sampling_rate = sampling_rate
        self.connect_arduino()
        
        # Real-time processing setup
        self.eeg_buffer = deque(maxlen=int(2.5 * sampling_rate))  # 2.5 seconds
        self.prediction_history = deque(maxlen=5)
        
        # Control parameters
        self.confidence_threshold = 0.65  # Lower threshold for 2-channel data
        self.last_command = "REST"
        self.is_running = False
        
        # Signal processing
        self.setup_filters()
        
        # Statistics
        self.sample_count = 0
        self.prediction_count = 0
        self.command_count = {'REST': 0, 'LEFT': 0, 'RIGHT': 0}
        
    def load_models(self):
        """Load trained BCI models"""
        try:
            print("📥 Loading BCI models...")
            
            self.model = joblib.load('bci_best_model.pkl')
            self.scaler = joblib.load('bci_feature_scaler.pkl')
            
            with open('bci_model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            print(f"✅ Model: {self.model_info['best_model']}")
            print(f"✅ Accuracy: {self.model_info['best_accuracy']:.1%}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
        
        return True
    
    def connect_arduino(self):
        """Connect to Arduino"""
        try:
            self.arduino = serial.Serial(
                self.arduino_port, 
                115200,  # High baud rate for EEG data
                timeout=1
            )
            time.sleep(3)  # Arduino boot time
            
            # Test connection
            self.arduino.write(b"STATUS\n")
            time.sleep(0.5)
            
            # Read response
            response = ""
            while self.arduino.in_waiting:
                response += self.arduino.readline().decode('utf-8', errors='ignore')
            
            if "SYSTEM STATUS" in response:
                print(f"✅ Arduino connected on {self.arduino_port}")
                self.arduino_connected = True
                
                # Test servo movements
                self.test_servos()
                return True
            else:
                print(f"⚠️ Arduino connected but no proper response")
                self.arduino_connected = True
                return True
            
        except Exception as e:
            print(f"❌ Arduino connection failed: {e}")
            print("Available ports: COM1, COM3, COM4, COM5...")
            self.arduino_connected = False
            return False
    
    def setup_filters(self):
        """Setup digital filters for EEG processing"""
        # Convert Arduino ADC values (0-1023) to approximate voltage
        self.adc_to_voltage = 5.0 / 1023.0
        
        # Bandpass filter (1-50 Hz)
        nyquist = self.sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = min(45.0, nyquist * 0.9) / nyquist
        
        self.b_bp, self.a_bp = signal.butter(4, [low_freq, high_freq], btype='band')
        
        # Notch filter for 50Hz power line interference
        notch_freq = 50.0 / nyquist
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, 30)
        
        print(f"✅ Filters ready: 1-45 Hz bandpass, 50Hz notch")
    
    def start_eeg_streaming(self):
        """Start EEG data streaming from Arduino"""
        if not self.arduino_connected:
            return False
        
        try:
            # Clear any existing data
            self.arduino.flushInput()
            
            # Start streaming
            self.arduino.write(b"STREAM_ON\n")
            time.sleep(0.5)
            
            print("✅ EEG streaming started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start streaming: {e}")
            return False
    
    def stop_eeg_streaming(self):
        """Stop EEG streaming"""
        if not self.arduino_connected:
            return
        
        try:
            self.arduino.write(b"STREAM_OFF\n")
            time.sleep(0.5)
            print("⏹️ EEG streaming stopped")
            
        except Exception as e:
            print(f"❌ Error stopping stream: {e}")
    
    def read_eeg_sample(self):
        """Read single EEG sample from Arduino"""
        if not self.arduino_connected:
            return None
        
        try:
            if self.arduino.in_waiting > 0:
                line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                
                if line and ',' in line:
                    parts = line.split(',')
                    
                    if len(parts) >= 3:  # timestamp,ch1,ch2
                        try:
                            timestamp = int(parts[0])
                            ch1 = int(parts[1])
                            ch2 = int(parts[2])
                            
                            # Convert ADC values to voltage (optional)
                            ch1_volt = ch1 * self.adc_to_voltage
                            ch2_volt = ch2 * self.adc_to_voltage
                            
                            return np.array([ch1_volt, ch2_volt])
                            
                        except ValueError:
                            return None
            
            return None
            
        except Exception as e:
            print(f"⚠️ EEG read error: {e}")
            return None
    
    def process_eeg_sample(self, eeg_sample):
        """Process EEG sample and add to buffer"""
        if eeg_sample is not None:
            self.eeg_buffer.append(eeg_sample)
            self.sample_count += 1
            
            # Extract features when we have enough data
            if len(self.eeg_buffer) >= int(1.5 * self.sampling_rate):
                return self.extract_features()
        
        return None
    
    def extract_features(self):
        """Extract features from EEG buffer"""
        if len(self.eeg_buffer) < int(self.sampling_rate):
            return None
        
        try:
            # Convert to numpy array
            eeg_window = np.array(list(self.eeg_buffer))
            
            # Apply filtering
            filtered_eeg = self.apply_filters(eeg_window)
            
            # Compute features
            features = self.compute_features(filtered_eeg)
            
            # Expand to match training data
            features_expanded = self.expand_features(features)
            
            return features_expanded
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    def apply_filters(self, eeg_data):
        """Apply digital filters"""
        try:
            # Remove DC component
            eeg_data = eeg_data - np.mean(eeg_data, axis=0)
            
            # Bandpass filter
            filtered = signal.filtfilt(self.b_bp, self.a_bp, eeg_data, axis=0)
            
            # Notch filter
            filtered = signal.filtfilt(self.b_notch, self.a_notch, filtered, axis=0)
            
            return filtered
            
        except Exception as e:
            print(f"⚠️ Filtering error: {e}")
            return eeg_data
    
    def compute_features(self, eeg_window):
        """Compute features from 2-channel EEG"""
        features = []
        
        # Time domain features for each channel
        for ch in range(2):
            channel_data = eeg_window[:, ch]
            
            # Statistical features
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.max(channel_data) - np.min(channel_data),  # Peak-to-peak
                np.percentile(channel_data, 75) - np.percentile(channel_data, 25),  # IQR
                np.sqrt(np.mean(channel_data**2)),  # RMS
                np.mean(np.abs(channel_data))  # Mean absolute value
            ])
        
        # Cross-channel features
        correlation = np.corrcoef(eeg_window[:, 0], eeg_window[:, 1])[0, 1]
        features.append(correlation if not np.isnan(correlation) else 0.0)
        
        # Frequency domain features
        try:
            for ch in range(2):
                channel_data = eeg_window[:, ch]
                
                # Power spectral density
                freqs, psd = signal.welch(channel_data, fs=self.sampling_rate, nperseg=128)
                
                # Band powers
                bands = {
                    'delta': (0.5, 4),
                    'theta': (4, 8), 
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 45)
                }
                
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        band_power = np.mean(psd[band_mask])
                        features.append(band_power)
                    else:
                        features.append(0.0)
        
        except Exception as e:
            print(f"⚠️ Frequency analysis error: {e}")
            features.extend([0.0] * 10)  # 5 bands × 2 channels
        
        return np.array(features)
    
    def expand_features(self, features):
        """Expand 2-channel features to match training size"""
        expected_size = self.model_info['feature_shape']
        current_size = len(features)
        
        if current_size < expected_size:
            # Intelligently repeat features
            repeat_factor = expected_size // current_size
            remainder = expected_size % current_size
            
            expanded = np.tile(features, repeat_factor)
            if remainder > 0:
                expanded = np.concatenate([expanded, features[:remainder]])
            
            return expanded
        
        elif current_size > expected_size:
            return features[:expected_size]
        
        return features
    
    def predict_movement(self, features):
        """Make BCI prediction"""
        if features is None:
            return "REST", 0.0
        
        try:
            features_reshaped = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_reshaped)
            
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            command_map = {0: "REST", 1: "LEFT", 2: "RIGHT"}
            command = command_map.get(prediction, "REST")
            
            return command, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "REST", 0.0
    
    def smooth_predictions(self, command):
        """Smooth predictions over time"""
        self.prediction_history.append(command)
        
        if len(self.prediction_history) >= 3:
            recent = list(self.prediction_history)[-3:]
            command_counts = {}
            for cmd in recent:
                command_counts[cmd] = command_counts.get(cmd, 0) + 1
            
            return max(command_counts.keys(), key=lambda x: command_counts[x])
        
        return command
    
    def send_servo_command(self, command):
        """Send servo command to Arduino"""
        if not self.arduino_connected:
            print(f"🤖 Simulated: {command}")
            return
        
        try:
            # Temporarily stop streaming for command
            self.arduino.write(b"STREAM_OFF\n")
            time.sleep(0.1)
            
            # Send servo command
            self.arduino.write(f"{command}\n".encode())
            time.sleep(0.2)
            
            # Resume streaming
            self.arduino.write(b"STREAM_ON\n")
            time.sleep(0.1)
            
            print(f"📡 Servo: {command}")
            
        except Exception as e:
            print(f"❌ Servo command error: {e}")
    
    def test_servos(self):
        """Test servo movements"""
        print("🔧 Testing servos...")
        test_commands = ["CENTER", "LEFT", "CENTER", "RIGHT", "CENTER"]
        
        for cmd in test_commands:
            self.arduino.write(f"{cmd}\n".encode())
            time.sleep(0.8)
        
        print("✅ Servo test complete")
    
    def realtime_bci_loop(self, duration=60):
        """Main real-time BCI processing loop"""
        print(f"\n🧠 Starting real-time BCI ({duration}s)...")
        print("Reading live EEG from Arduino BioAmp...")
        print("Press Ctrl+C to stop early")
        print("-" * 60)
        
        # Start EEG streaming
        if not self.start_eeg_streaming():
            return False
        
        start_time = time.time()
        last_prediction_time = 0
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Read EEG sample
                eeg_sample = self.read_eeg_sample()
                
                if eeg_sample is not None:
                    # Process sample
                    features = self.process_eeg_sample(eeg_sample)
                    
                    # Make prediction every 0.5 seconds
                    if features is not None and (time.time() - last_prediction_time) > 0.5:
                        command, confidence = self.predict_movement(features)
                        smoothed_command = self.smooth_predictions(command)
                        
                        # Act on high-confidence predictions
                        if confidence > self.confidence_threshold:
                            final_command = smoothed_command
                        else:
                            final_command = "REST"
                        
                        # Send command if changed
                        if final_command != self.last_command:
                            self.send_servo_command(final_command)
                            self.last_command = final_command
                            self.command_count[final_command] += 1
                        
                        # Display status
                        elapsed = time.time() - start_time
                        self.prediction_count += 1
                        
                        print(f"Time: {elapsed:5.1f}s | Samples: {self.sample_count:5d} | "
                              f"EEG→{command:5s} | Conf: {confidence:.2f} | "
                              f"Action: {final_command:5s}")
                        
                        last_prediction_time = time.time()
                
                # Small delay
                time.sleep(0.002)  # 2ms
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        except Exception as e:
            print(f"\n❌ Processing error: {e}")
        
        finally:
            # Cleanup
            self.stop_eeg_streaming()
            self.send_servo_command("CENTER")
            
            # Show statistics
            total_time = time.time() - start_time
            print(f"\n📊 Session Statistics:")
            print(f"Duration: {total_time:.1f} seconds")
            print(f"EEG samples: {self.sample_count:,}")
            print(f"Sample rate: {self.sample_count/total_time:.1f} Hz")
            print(f"Predictions: {self.prediction_count}")
            print(f"Commands sent:")
            for cmd, count in self.command_count.items():
                print(f"  {cmd}: {count}")
            
            print("✅ BCI session complete!")
    
    def interactive_mode(self):
        """Interactive control mode"""
        print("\n🎮 Interactive BCI Mode")
        print("Commands:")
        print("  'r' - Start real-time BCI (30s)")
        print("  't' - Test servos")
        print("  's' - Show EEG stream (10s)")
        print("  'c' - Calibrate (read baseline)")
        print("  'q' - Quit")
        
        while True:
            try:
                cmd = input("\nEnter command: ").lower().strip()
                
                if cmd == 'q':
                    break
                elif cmd == 'r':
                    self.is_running = True
                    self.realtime_bci_loop(30)
                    self.is_running = False
                elif cmd == 't':
                    self.test_servos()
                elif cmd == 's':
                    self.show_eeg_stream(10)
                elif cmd == 'c':
                    self.calibrate_baseline()
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                break
        
        # Cleanup
        if self.arduino_connected:
            self.arduino.close()
        print("👋 BCI system shutdown")
    
    def show_eeg_stream(self, duration=10):
        """Show raw EEG data stream"""
        print(f"📊 Showing EEG stream for {duration}s...")
        
        if not self.start_eeg_streaming():
            return
        
        start_time = time.time()
        samples = []
        
        try:
            while (time.time() - start_time) < duration: