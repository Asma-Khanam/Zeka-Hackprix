import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import joblib
import time
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os
import mne
from mne.datasets import eegbci
import threading
from scipy import signal as sg
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

COLORS = {
    'background': '#f0f0f0',
    'panel_bg': '#ffffff',
    'text': '#333333',
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'accent': '#FF9800',
    'error': '#F44336',
    'grid': '#dddddd',
    'states': {
        'rest': '#4CAF50',
        'left': '#2196F3',
        'right': '#FF9800'
    }
}

default_channels = ['C3', 'Cz', 'C4']
class_names = ['rest', 'left', 'right']
task_runs = {
    'Task4': [4], # Handle multiple runs per task if needed (the list format [4] could be [4, 6, 10] for multiple runs)
    'Task5': [8]
}

os.makedirs('models', exist_ok=True)

def download_physionet_data(subject_num, run):
    print(f"Downloading data for subject {subject_num}, run {run}...")
    os.makedirs('files/MNE-eegbci-data/files/eegmmidb/1.0.0', exist_ok=True)
    
    try:
        eegbci.load_data(subject_num, runs=[run], path='files/')
        file_path = os.path.join('files', f'MNE-eegbci-data', f'files', f'eegmmidb', f'1.0.0',
                              f'S{str(subject_num).zfill(3)}', 
                              f'S{str(subject_num).zfill(3)}R{str(run).zfill(2)}.edf')
        
        if os.path.exists(file_path):
            print(f"Successfully downloaded data: {file_path}")
            return True, file_path
        else:
            print(f"Download completed but file not found: {file_path}")
            return False, None
    except Exception as e:
        print(f"Download failed: {e}")
        return False, None

def create_rounded_frame(parent, **kwargs):
    frame = ttk.Frame(parent, **kwargs)
    return frame

def extract_advanced_features(epochs_data, fs=160):
    if isinstance(epochs_data, np.ndarray):
        # Handle raw numpy array
        n_channels = epochs_data.shape[0]
        n_times = epochs_data.shape[1]
        n_epochs = 1
        data = epochs_data.reshape(1, n_channels, n_times)
    else:
        # Handle MNE Epochs object
        n_epochs, n_channels, n_times = epochs_data.get_data().shape
        data = epochs_data.get_data()
    
    freq_bands = [
        (4, 8),  # Theta
        (8, 10), # Low Alpha  
        (10, 13), # High Alpha
        (13, 20), # Low Beta
        (20, 30), # High Beta
    ]
# Uses Welch's method to compute power spectral density
# Calculates average power in each frequency band
    
    n_features = n_channels * (len(freq_bands) + 3) + 20
    X_features = np.zeros((n_epochs, n_features))
    
    for epoch_idx in range(n_epochs):
        feature_idx = 0
        for ch_idx in range(n_channels):
            if np.isnan(data[epoch_idx, ch_idx]).any() or np.isinf(data[epoch_idx, ch_idx]).any():
                clean_data = np.nan_to_num(data[epoch_idx, ch_idx], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                clean_data = data[epoch_idx, ch_idx]
                
            try:
                freqs, psd = sg.welch(clean_data, fs=fs, nperseg=min(512, n_times))
                
                for low_freq, high_freq in freq_bands:
                    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                    if np.any(idx_band):
                        band_power = np.mean(psd[idx_band])
                        if np.isnan(band_power) or np.isinf(band_power):
                            band_power = 0
                    else:
                        band_power = 0
                    X_features[epoch_idx, feature_idx] = band_power
                    feature_idx += 1
            except Exception as e:
                print(f"Error in spectral feature calculation: {e}")
                for _ in range(len(freq_bands)):
                    X_features[epoch_idx, feature_idx] = 0
                    feature_idx += 1
            
            try:
                var_val = np.var(clean_data)
                X_features[epoch_idx, feature_idx] = 0 if np.isnan(var_val) else var_val
                feature_idx += 1
                
                if np.std(clean_data) > 1e-10:
                    skew_val = stats.skew(clean_data)
                    X_features[epoch_idx, feature_idx] = 0 if np.isnan(skew_val) else skew_val
                else:
                    X_features[epoch_idx, feature_idx] = 0
                feature_idx += 1
                
                if np.std(clean_data) > 1e-10:
                    kurt_val = stats.kurtosis(clean_data)
                    X_features[epoch_idx, feature_idx] = 0 if np.isnan(kurt_val) else kurt_val
                else:
                    X_features[epoch_idx, feature_idx] = 0
                feature_idx += 1
            except Exception as e:
                print(f"Error in temporal feature calculation: {e}")
                for _ in range(3):
                    X_features[epoch_idx, feature_idx] = 0
                    feature_idx += 1
    
    motor_ch_indices = list(range(min(3, n_channels)))
    
    if len(motor_ch_indices) >= 2:
        for i, ch1 in enumerate(motor_ch_indices):
            for j, ch2 in enumerate(motor_ch_indices):
                if i < j and feature_idx < n_features:
                    for epoch_idx in range(n_epochs):
                        try:
                            data1 = data[epoch_idx, ch1]
                            data2 = data[epoch_idx, ch2]
                            
                            if (np.isnan(data1).any() or np.isinf(data1).any() or 
                                np.isnan(data2).any() or np.isinf(data2).any()):
                                corr = 0
                            else:
                                if np.std(data1) > 1e-10 and np.std(data2) > 1e-10:
                                    corr = np.corrcoef(data1, data2)[0, 1]
                                    if np.isnan(corr) or np.isinf(corr):
                                        corr = 0
                                else:
                                    corr = 0
                            X_features[epoch_idx, feature_idx] = corr
                        except Exception as e:
                            print(f"Error in connectivity calculation: {e}")
                            X_features[epoch_idx, feature_idx] = 0
                    feature_idx += 1
    
    n_spectral = n_channels * len(freq_bands)
    X_features[:, :n_spectral] = np.where(X_features[:, :n_spectral] > 0, 
                                         np.log(X_features[:, :n_spectral] + 1e-10),
                                         0)
    
    if feature_idx < n_features:
        X_features[:, feature_idx:] = 0
    
    if np.isnan(X_features).any() or np.isinf(X_features).any():
        print("Warning: NaN or Inf values found in features after extraction")
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if n_epochs == 1:
        return X_features[0]
    else:
        return X_features

def load_or_train_model(X=None, y=None):
    model_path = 'models/high_acc_stack_model.pkl'
    
    try:
        model = joblib.load(model_path)
        print(f"Model type: {type(model)}")
        print(f"Model contents: {model.keys() if isinstance(model, dict) else 'Not a dict'}")
        return model, True
    except Exception as e:
        print(f"Could not load model: {e}")
        
        if X is not None and y is not None and len(X) > 0 and len(y) > 0:
            print("Training new model with provided data...")
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                ))
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model.fit(X_train, y_train)
            
            accuracy = model.score(X_test, y_test)
            print(f"Model trained with accuracy: {accuracy:.2f}")
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            try:
                joblib.dump(model, model_path)
                print(f"Model saved to {model_path}")
            except Exception as save_err:
                print(f"Could not save model: {save_err}")
            
            return model, True
        else:
            print("No data provided for training. Please load data first.")
            return None, False

class DynamicBCIVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Brain-Computer Interface | Real-time Motor Imagery Analysis")
        self.root.geometry("1300x900")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('TFrame', background=COLORS['background'])
        self.style.configure('TLabel', background=COLORS['background'], foreground=COLORS['text'])
        self.style.configure('Header.TLabel', font=('Helvetica', 24, 'bold'), foreground=COLORS['primary'])
        self.style.configure('Subheader.TLabel', font=('Helvetica', 16), foreground=COLORS['text'])
        self.style.configure('StateBox.TFrame', borderwidth=2, relief='raised')
        self.style.configure('ActiveState.TFrame', borderwidth=2, relief='raised', background=COLORS['primary'])
        self.style.configure('TNotebook', background=COLORS['background'])
        self.style.configure('TNotebook.Tab', background=COLORS['background'], foreground=COLORS['text'],
                           font=('Helvetica', 12))
        self.style.map('TNotebook.Tab', background=[('selected', COLORS['primary'])],
                     foreground=[('selected', 'white')])
        
        self.data_loaded = False
        self.model_loaded = False
        self.model = None
        self.subjects = [i for i in range(1, 11)]
        self.current_subject = 1
        self.current_task = 'Task4'
        self.raw_data = None
        self.epochs = None
        self.current_epoch = 0
        self.epoch_data = None
        self.available_channels = []
        self.selected_channels = []
        self.fs = 160
        self.channel_names = []
        self.time_axis = None
        
        self.features = None
        self.feature_vector = None
        self.alpha_powers = None
        self.beta_powers = None
        self.spectrogram_data = None
        
        self.buffer_size = 500
        self.signals_buffer = None
        
        self.auto_advance = True
        self.advance_interval = 3000
        self.paused = False
        
        self.animation_speed = 100
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.title_label = ttk.Label(
            self.main_frame, 
            text="Dynamic Brain-Computer Interface: Real-time Motor Imagery Analysis",
            style='Header.TLabel'
        )
        self.title_label.pack(pady=10)
        
        self.data_selector_frame = create_rounded_frame(self.main_frame)
        self.data_selector_frame.pack(fill=tk.X, pady=5)
        
        self.subject_label = ttk.Label(
            self.data_selector_frame,
            text="Subject:",
            font=("Helvetica", 12)
        )
        self.subject_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.subject_var = tk.StringVar(value="1")
        self.subject_menu = ttk.Combobox(
            self.data_selector_frame,
            textvariable=self.subject_var,
            values=[str(i) for i in self.subjects],
            width=5
        )
        self.subject_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.subject_menu.bind("<<ComboboxSelected>>", self.on_subject_change)
        
        self.task_label = ttk.Label(
            self.data_selector_frame,
            text="Task:",
            font=("Helvetica", 12)
        )
        self.task_label.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        self.task_var = tk.StringVar(value="Task4")
        self.task_menu = ttk.Combobox(
            self.data_selector_frame,
            textvariable=self.task_var,
            values=list(task_runs.keys()),
            width=8
        )
        self.task_menu.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.task_menu.bind("<<ComboboxSelected>>", self.on_task_change)
        
        self.load_button = ttk.Button(
            self.data_selector_frame,
            text="Load Data",
            command=self.load_data
        )
        self.load_button.grid(row=0, column=4, padx=10, pady=5, sticky="w")
        
        self.train_button = ttk.Button(
            self.data_selector_frame,
            text="Train Model",
            command=self.train_model,
            state='disabled'
        )
        self.train_button.grid(row=0, column=5, padx=10, pady=5, sticky="w")
        
        self.info_frame = create_rounded_frame(self.main_frame)
        self.info_frame.pack(fill=tk.X, pady=5)
        
        self.current_state = tk.StringVar()
        self.current_state.set("NOT LOADED")
        
        self.state_label = ttk.Label(
            self.info_frame,
            text="Mental Command:",
            font=("Helvetica", 12, "bold"),
        )
        self.state_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.state_value = ttk.Label(
            self.info_frame,
            textvariable=self.current_state,
            font=("Helvetica", 14, "bold"),
            foreground=COLORS['secondary']
        )
        self.state_value.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Status: Ready to load real EEG data")
        self.status_label = ttk.Label(
            self.info_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10)
        )
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tab frames
        self.tab1 = ttk.Frame(self.notebook)  # EEG Signals and Spectrogram
        self.tab2 = ttk.Frame(self.notebook)  # Feature Analysis
        self.tab3 = ttk.Frame(self.notebook)  # Brain Topography
        
        # Add tabs to notebook
        self.notebook.add(self.tab1, text="EEG Signals & Spectrogram")
        self.notebook.add(self.tab2, text="Feature Analysis")
        self.notebook.add(self.tab3, text="Brain Topography")
        
        # Setup visualizations in appropriate tabs
        self.setup_eeg_visualization()
        self.setup_timefreq_visualization()
        self.setup_feature_visualization()
        self.setup_topography_visualization()
        
        self.control_frame = create_rounded_frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)
        
        self.navigation_frame = ttk.Frame(self.control_frame)
        self.navigation_frame.pack(pady=10)
        
        self.prev_button = ttk.Button(
            self.navigation_frame,
            text="◄ Prev Epoch",
            command=self.prev_epoch,
            state='disabled'
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.playpause_button = ttk.Button(
            self.navigation_frame,
            text="❚❚ Pause",
            command=self.toggle_play_pause,
            state='disabled'
        )
        self.playpause_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(
            self.navigation_frame,
            text="Next Epoch ►",
            command=self.next_epoch,
            state='disabled'
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.auto_advance_var = tk.BooleanVar(value=True)
        self.auto_advance_check = ttk.Checkbutton(
            self.navigation_frame,
            text="Auto-advance",
            variable=self.auto_advance_var,
            command=self.toggle_auto_advance
        )
        self.auto_advance_check.pack(side=tk.LEFT, padx=20)
        
        self.epoch_var = tk.StringVar(value="Epoch: -/-")
        self.epoch_label = ttk.Label(
            self.navigation_frame,
            textvariable=self.epoch_var,
            font=("Helvetica", 10)
        )
        self.epoch_label.pack(side=tk.LEFT, padx=20)
        
        self.states_frame = ttk.Frame(self.control_frame)
        self.states_frame.pack(pady=10)
        
        self.state_boxes = {}
        
        for i, state in enumerate(class_names):
            state_frame = ttk.Frame(self.states_frame, width=100, height=60, style='StateBox.TFrame')
            state_frame.pack(side=tk.LEFT, padx=15)
            state_frame.pack_propagate(False)
            
            state_label = ttk.Label(
                state_frame,
                text=state.upper(),
                font=("Helvetica", 14, "bold"),
                anchor="center",
            )
            state_label.pack(expand=True, fill=tk.BOTH)
            
            self.state_boxes[state] = (state_frame, state_label)
        
        self.auto_advance_id = None
        self._update_cache = {}
        self.anims = []
        self.start_animations()
        self.load_model()
    
    def setup_eeg_visualization(self):
        # Create a frame within tab1 for EEG signals
        self.eeg_frame = ttk.LabelFrame(self.tab1, text="EEG Signals")
        self.eeg_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.eeg_fig = plt.Figure(figsize=(6, 4.5), dpi=100)
        self.eeg_gs = gridspec.GridSpec(3, 1)
        
        self.eeg_canvas = FigureCanvasTkAgg(self.eeg_fig, self.eeg_frame)
        self.eeg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.eeg_axes = []
        self.highlight_boxes = []
        self.erd_annotations = []
        
        channel_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        
        for i in range(3):
            ax = self.eeg_fig.add_subplot(self.eeg_gs[i])
            ax.set_facecolor(COLORS['panel_bg'])
            ax.set_title(default_channels[i], fontsize=12, color=COLORS['text'], fontweight='bold')
            ax.set_ylabel('Amplitude (μV)', fontsize=10, color=COLORS['text'])
            
            if i == 2:
                ax.set_xlabel('Time (seconds)', fontsize=10, color=COLORS['text'])
            
            ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
            
            for spine in ax.spines.values():
                spine.set_color(COLORS['grid'])
                spine.set_linewidth(0.5)
            
            line, = ax.plot(
                np.arange(self.buffer_size), 
                np.zeros(self.buffer_size), 
                color=channel_colors[i],
                linewidth=1.5,
                alpha=0.8
            )
            
            highlight = ax.add_patch(Rectangle((0, 0), 0, 0, alpha=0.2, color='none'))
            self.highlight_boxes.append(highlight)
            
            erd_text = ax.text(
                0.5, 0.5, '', 
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
                visible=False
            )
            self.erd_annotations.append(erd_text)
            
            self.eeg_axes.append((ax, line))
        
        explanation_frame = ttk.Frame(self.eeg_frame)
        explanation_frame.pack(pady=5, fill=tk.X)
        
        bands_frame = ttk.Frame(explanation_frame)
        bands_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(bands_frame, text="Frequency Bands:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        ttk.Label(bands_frame, text="Delta (0.5-4 Hz)", foreground="#9C27B0").pack(side=tk.LEFT, padx=5)
        ttk.Label(bands_frame, text="Theta (4-8 Hz)", foreground="#FF9800").pack(side=tk.LEFT, padx=5)
        ttk.Label(bands_frame, text="Alpha (8-13 Hz)", foreground="#4CAF50").pack(side=tk.LEFT, padx=5)
        ttk.Label(bands_frame, text="Beta (13-30 Hz)", foreground="#2196F3").pack(side=tk.LEFT, padx=5)
        
        pattern_frame = ttk.Frame(explanation_frame)
        pattern_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(pattern_frame, text="ERD:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        ttk.Label(pattern_frame, text="Event-Related Desynchronization", foreground="#F44336").pack(side=tk.LEFT, padx=5)
        
        self.eeg_fig.tight_layout()
    
    def setup_timefreq_visualization(self):
        # Create a frame within tab1 for time-frequency analysis
        self.tf_frame = ttk.LabelFrame(self.tab1, text="Time-Frequency Analysis")
        self.tf_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.tf_fig = plt.Figure(figsize=(6, 3), dpi=100)
        self.tf_canvas = FigureCanvasTkAgg(self.tf_fig, self.tf_frame)
        self.tf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.tf_ax = self.tf_fig.add_subplot(111)
        self.tf_ax.set_facecolor(COLORS['panel_bg'])
        self.tf_ax.set_title('Spectrogram (C3 Channel)', fontsize=12, color=COLORS['text'])
        self.tf_ax.set_ylabel('Frequency (Hz)', fontsize=10, color=COLORS['text'])
        self.tf_ax.set_xlabel('Time (seconds)', fontsize=10, color=COLORS['text'])
        
        spec_data = np.zeros((30, 100))
        self.spectrogram = self.tf_ax.imshow(
            spec_data,
            aspect='auto',
            origin='lower',
            extent=[0, 5, 0, 30],
            cmap='viridis',
            interpolation='bilinear'
        )
        
        cbar = self.tf_fig.colorbar(self.spectrogram, ax=self.tf_ax, shrink=0.8)
        cbar.set_label('Power', fontsize=8, color=COLORS['text'])
        
        self.tf_ax.axhline(y=4, color='white', linestyle='--', alpha=0.5)
        self.tf_ax.axhline(y=8, color='white', linestyle='--', alpha=0.5)
        self.tf_ax.axhline(y=13, color='white', linestyle='--', alpha=0.5)
        self.tf_ax.axhline(y=30, color='white', linestyle='--', alpha=0.5)
        
        self.tf_ax.text(5.1, 2, 'Delta', fontsize=8, ha='left', va='center', color='white')
        self.tf_ax.text(5.1, 6, 'Theta', fontsize=8, ha='left', va='center', color='white')
        self.tf_ax.text(5.1, 10, 'Alpha', fontsize=8, ha='left', va='center', color='white')
        self.tf_ax.text(5.1, 20, 'Beta', fontsize=8, ha='left', va='center', color='white')
        
        explanation_frame = ttk.Frame(self.tf_frame)
        explanation_frame.pack(pady=5, fill=tk.X)
        
        explanation_text = (
            "Spectrogram shows how the frequency content changes over time.\n"
            "For motor imagery, look for alpha band (8-13 Hz) power decrease (ERD)."
        )
        ttk.Label(explanation_frame, text=explanation_text, font=("Helvetica", 9)).pack()
        
        self.tf_fig.tight_layout()
    
    def setup_feature_visualization(self):
        # Create a frame within tab2 for feature analysis
        self.feature_frame = ttk.LabelFrame(self.tab2, text="EEG Feature Analysis")
        self.feature_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.feature_fig = plt.Figure(figsize=(6, 4.5), dpi=100)
        self.feature_canvas = FigureCanvasTkAgg(self.feature_fig, self.feature_frame)
        self.feature_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        gs = gridspec.GridSpec(2, 2, figure=self.feature_fig)
        
        self.radar_ax = self.feature_fig.add_subplot(gs[0, 0], polar=True)
        self.radar_ax.set_facecolor(COLORS['panel_bg'])
        self.radar_ax.set_title('EEG Band Power Distribution', fontsize=12, color=COLORS['text'])
        
        self.radar_categories = ['C3-Theta', 'C3-Alpha', 'C3-Beta', 
                                'Cz-Theta', 'Cz-Alpha', 'Cz-Beta',
                                'C4-Theta', 'C4-Alpha', 'C4-Beta']
        
        self.radar_num_vars = len(self.radar_categories)
        self.radar_angles = np.linspace(0, 2*np.pi, self.radar_num_vars, endpoint=False).tolist()
        self.radar_angles += self.radar_angles[:1]
        
        self.radar_ax.set_xticks(self.radar_angles[:-1])
        self.radar_ax.set_xticklabels(self.radar_categories, color=COLORS['text'], fontsize=7)
        
        self.radar_values = [0] * self.radar_num_vars
        self.radar_values += self.radar_values[:1]
        
        self.radar_line, = self.radar_ax.plot(self.radar_angles, self.radar_values, 
                                              linewidth=2, linestyle='solid', 
                                              color=COLORS['primary'])
        self.radar_fill = self.radar_ax.fill(self.radar_angles, self.radar_values, 
                           color=COLORS['primary'], alpha=0.25)
        
        self.bar_ax = self.feature_fig.add_subplot(gs[0, 1])
        self.bar_ax.set_facecolor(COLORS['panel_bg'])
        self.bar_ax.set_title('Band Power Comparison', fontsize=12, color=COLORS['text'])
        self.bar_ax.set_xlabel('Channel-Band', fontsize=9, color=COLORS['text'])
        self.bar_ax.set_ylabel('Power', fontsize=9, color=COLORS['text'])
        
        self.bar_x = np.arange(len(self.radar_categories))
        
        self.bars = self.bar_ax.bar(
            self.bar_x, 
            np.zeros(len(self.radar_categories)),
            color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']] * 3,
            alpha=0.7
        )
        
        self.bar_ax.set_xticks(self.bar_x)
        self.bar_ax.set_xticklabels(self.radar_categories, rotation=45, ha='right', fontsize=7, color=COLORS['text'])
        
        self.bar_ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
        
        self.timeseries_ax = self.feature_fig.add_subplot(gs[1, 0])
        self.timeseries_ax.set_facecolor(COLORS['panel_bg'])
        self.timeseries_ax.set_title('Alpha Power Over Time', fontsize=12, color=COLORS['text'])
        self.timeseries_ax.set_xlabel('Time (seconds)', fontsize=9, color=COLORS['text'])
        self.timeseries_ax.set_ylabel('Alpha Power', fontsize=9, color=COLORS['text'])
        
        self.ts_buffer_size = 100
        self.ts_time = np.linspace(-10, 0, self.ts_buffer_size)
        self.ts_c3_data = np.zeros(self.ts_buffer_size)
        self.ts_cz_data = np.zeros(self.ts_buffer_size)
        self.ts_c4_data = np.zeros(self.ts_buffer_size)
        
        self.ts_c3_line, = self.timeseries_ax.plot(self.ts_time, self.ts_c3_data, 
                                                   linewidth=1.5, color=COLORS['primary'], 
                                                   label='C3 (Left)')
        self.ts_cz_line, = self.timeseries_ax.plot(self.ts_time, self.ts_cz_data, 
                                                  linewidth=1.5, color=COLORS['secondary'], 
                                                  label='Cz (Center)')
        self.ts_c4_line, = self.timeseries_ax.plot(self.ts_time, self.ts_c4_data, 
                                                   linewidth=1.5, color=COLORS['accent'], 
                                                   label='C4 (Right)')
        
        self.timeseries_ax.legend(fontsize=8)
        
        self.timeseries_ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
        
        self.erd_ax = self.feature_fig.add_subplot(gs[1, 1])
        self.erd_ax.set_facecolor(COLORS['panel_bg'])
        self.erd_ax.set_title('Event-Related Desynchronization (ERD)', fontsize=12, color=COLORS['text'])
        
        self.erd_x = np.linspace(0, 5, 500)
        
        self.baseline = np.sin(2*np.pi*10*self.erd_x) * 0.8 + np.random.normal(0, 0.1, 500)
        self.erd_signal = np.copy(self.baseline)
        
        erd_idx = np.logical_and(self.erd_x >= 2, self.erd_x <= 3)
        self.erd_signal[erd_idx] = self.erd_signal[erd_idx] * 0.4
        
        self.erd_baseline_line, = self.erd_ax.plot(self.erd_x, self.baseline, color='gray', alpha=0.5, linewidth=1, label='Baseline')
        self.erd_signal_line, = self.erd_ax.plot(self.erd_x, self.erd_signal, color=COLORS['primary'], linewidth=1.5, label='ERD')
        
        self.erd_highlight = self.erd_ax.axvspan(2, 3, alpha=0.2, color=COLORS['primary'])
        self.erd_text = self.erd_ax.text(2.5, 0, 'ERD', ha='center', fontsize=10, color=COLORS['text'],
                       bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=COLORS['primary'], alpha=0.7))
        
        self.erd_ax.text(0.5, -1.5, 
                       "ERD is a decrease in neural oscillations (particularly in alpha band)\n"
                       "when a brain region becomes active during motor imagery.",
                       ha='center', fontsize=8, color=COLORS['text'])
        
        self.erd_ax.legend(fontsize=8)
        
        self.erd_ax.set_xlim(0, 5)
        self.erd_ax.set_ylim(-2, 2)
        
        self.erd_ax.set_yticks([])
        
        self.erd_ax.set_xlabel('Time (s)', fontsize=9, color=COLORS['text'])
        
        self.feature_fig.tight_layout()
    
    def setup_topography_visualization(self):
        # Create a frame within tab3 for brain topography
        self.topo_frame = ttk.LabelFrame(self.tab3, text="Brain Topography")
        self.topo_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.topo_fig = plt.Figure(figsize=(6, 3), dpi=100)
        self.topo_canvas = FigureCanvasTkAgg(self.topo_fig, self.topo_frame)
        self.topo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        topo_gs = gridspec.GridSpec(1, 2, figure=self.topo_fig)
        
        self.topo_alpha_ax = self.topo_fig.add_subplot(topo_gs[0, 0])
        self.topo_alpha_ax.set_facecolor(COLORS['panel_bg'])
        self.topo_alpha_ax.set_title('Alpha Band (8-13 Hz)', fontsize=12, color=COLORS['text'])
        
        self.topo_beta_ax = self.topo_fig.add_subplot(topo_gs[0, 1])
        self.topo_beta_ax.set_facecolor(COLORS['panel_bg'])
        self.topo_beta_ax.set_title('Beta Band (13-30 Hz)', fontsize=12, color=COLORS['text'])
        
        self.draw_head_outline(self.topo_alpha_ax, view='top')
        self.draw_head_outline(self.topo_beta_ax, view='top')
        
        self.mark_electrodes(self.topo_alpha_ax, view='top')
        self.mark_electrodes(self.topo_beta_ax, view='top')
        
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        self.topo_cmap = LinearSegmentedColormap.from_list('ERD_ERS', colors, N=100)
        
        n = 20
        self.topo_grid_size = n
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        mask = X**2 + Y**2 <= 1
        
        self.topo_x = x
        self.topo_y = y
        self.topo_mask = mask
        
        data = np.zeros((n, n))
        data[mask] = np.random.rand(np.sum(mask))
        
        self.alpha_heatmap = self.topo_alpha_ax.imshow(
            data, 
            extent=[-1.2, 1.2, -1.2, 1.2],
            origin='lower',
            cmap=self.topo_cmap,
            vmin=-1,
            vmax=1,
            interpolation='bilinear'
        )
        
        self.beta_heatmap = self.topo_beta_ax.imshow(
            data, 
            extent=[-1.2, 1.2, -1.2, 1.2],
            origin='lower',
            cmap=self.topo_cmap,
            vmin=-1,
            vmax=1,
            interpolation='bilinear'
        )
        
        cbar_alpha = self.topo_fig.colorbar(self.alpha_heatmap, ax=self.topo_alpha_ax, shrink=0.6)
        cbar_alpha.set_label('Activity', fontsize=8, color=COLORS['text'])
        
        cbar_beta = self.topo_fig.colorbar(self.beta_heatmap, ax=self.topo_beta_ax, shrink=0.6)
        cbar_beta.set_label('Activity', fontsize=8, color=COLORS['text'])
        
        explanation_frame = ttk.Frame(self.topo_frame)
        explanation_frame.pack(pady=5, fill=tk.X)
        
        explanation_text = (
            "Brain topography shows activity distribution across the scalp. "
            "Red = higher activity, Blue = lower activity (ERD)"
        )
        ttk.Label(explanation_frame, text=explanation_text, font=("Helvetica", 9)).pack()
        
        self.topo_fig.tight_layout()
    
    def draw_head_outline(self, ax, view='top'):
        if view == 'top':
            circle = plt.Circle((0, 0), 1, fill=False, color=COLORS['text'], linewidth=2)
            ax.add_patch(circle)
            ax.plot([0, 0], [0.9, 1.1], color=COLORS['text'], linewidth=2)
            ax.plot([-1.1, -0.9], [0, 0], color=COLORS['text'], linewidth=2)
            ax.plot([0.9, 1.1], [0, 0], color=COLORS['text'], linewidth=2)
            
        elif view == 'side':
            x = np.linspace(-1, 1, 100)
            y_top = np.sqrt(1 - x**2)
            ax.plot(x, y_top, color=COLORS['text'], linewidth=2)
            ax.plot([-1, 1], [0, 0], color=COLORS['text'], linewidth=2)
            ax.plot([0.9, 1.1], [0.2, 0.2], color=COLORS['text'], linewidth=2)
            ax.plot([-1.05, -1.05], [-0.2, 0.2], color=COLORS['text'], linewidth=2)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    
    def mark_electrodes(self, ax, view='top'):
        if view == 'top':
            positions = {
                'C3': (-0.5, 0),
                'Cz': (0, 0),
                'C4': (0.5, 0),
                'F3': (-0.5, 0.5),
                'Fz': (0, 0.5),
                'F4': (0.5, 0.5),
                'P3': (-0.5, -0.5),
                'Pz': (0, -0.5),
                'P4': (0.5, -0.5)
            }
            
            for name, (x, y) in positions.items():
                ax.plot(x, y, 'o', markersize=8, color=COLORS['primary'])
                ax.text(x, y+0.1, name, ha='center', va='center', fontsize=8, color=COLORS['text'])
                
                if name in ['C3', 'Cz', 'C4']:
                    ax.plot(x, y, 'o', markersize=10, mfc='none', mec=COLORS['accent'], linewidth=2)
        
        elif view == 'side':
            positions = {
                'C3': (-0.5, 0.75),
                'Cz': (0, 1),
                'C4': (0.5, 0.75),
                'F3': (-0.75, 0.5),
                'Fz': (0, 0.75),
                'F4': (0.75, 0.5),
            }
            
            for name, (x, y) in positions.items():
                if x >= -0.2:
                    ax.plot(x, y, 'o', markersize=8, color=COLORS['primary'])
                    ax.text(x, y+0.1, name, ha='center', va='center', fontsize=8, color=COLORS['text'])
                    
                    if name in ['Cz', 'C4']:
                        ax.plot(x, y, 'o', markersize=10, mfc='none', mec=COLORS['accent'], linewidth=2)
    
    def start_animations(self):
        self.eeg_ani = animation.FuncAnimation(
            self.eeg_fig, self.update_eeg_plot, 
            interval=self.animation_speed,
            blit=False
        )
        self.anims.append(self.eeg_ani)
        
        self.feature_ani = animation.FuncAnimation(
            self.feature_fig, self.update_feature_plots, 
            interval=self.animation_speed*2,
            blit=False
        )
        self.anims.append(self.feature_ani)
        
        self.tf_ani = animation.FuncAnimation(
            self.tf_fig, self.update_timefreq_plot, 
            interval=self.animation_speed*2,
            blit=False
        )
        self.anims.append(self.tf_ani)
        
        self.topo_ani = animation.FuncAnimation(
            self.topo_fig, self.update_topography, 
            interval=self.animation_speed*2,
            blit=False
        )
        self.anims.append(self.topo_ani)
    
    def on_subject_change(self, event=None):
        try:
            self.current_subject = int(self.subject_var.get())
            self.data_loaded = False
            self.epoch_data = None
            self.status_var.set(f"Status: Selected Subject {self.current_subject}")
        except ValueError:
            pass
    
    def on_task_change(self, event=None):
        self.current_task = self.task_var.get()
        self.data_loaded = False
        self.epoch_data = None
        self.status_var.set(f"Status: Selected {self.current_task}")

    def load_model(self):
        try:
            model_path = 'models/high_acc_stack_model.pkl'
            os.makedirs('models', exist_ok=True)
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.model_loaded = True
                self.status_var.set(f"Status: Model loaded successfully from {model_path}")
            else:
                self.status_var.set("Status: Model not found. Will train after loading data.")
                self.model_loaded = False
        except Exception as e:
            self.status_var.set(f"Status: Error loading model: {e}")
            self.model_loaded = False
    
    def train_model(self):
        if not self.data_loaded or self.epochs is None:
            messagebox.showwarning("Training Error", "Please load EEG data first before training.")
            return
        
        self.status_var.set("Status: Training model...")
        self.root.update_idletasks()
        
        try:
            X = []
            y = []
            
            for i in range(len(self.epochs)):
                epoch_data = self.epochs[i].get_data()[0]
                
                event_id = self.epochs.events[i, 2]
                event_dict = {v: k for k, v in self.epochs.event_id.items()}
                state = event_dict.get(event_id, 'rest')
                
                if 'rest' in state.lower() or 't0' in state.lower():
                    label = 0
                elif 'left' in state.lower() or 't1' in state.lower():
                    label = 1
                elif 'right' in state.lower() or 't2' in state.lower():
                    label = 2
                else:
                    label = 0
                
                features = extract_advanced_features(epoch_data, self.fs)
                
                X.append(features)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            self.model, success = load_or_train_model(X, y)
            
            if success:
                self.model_loaded = True
                self.status_var.set("Status: Model trained and saved successfully")
                messagebox.showinfo("Training Complete", "Model was successfully trained and saved.")
            else:
                self.status_var.set("Status: Error training model")
                messagebox.showerror("Training Error", "Failed to train model.")
        
        except Exception as e:
            self.status_var.set(f"Status: Error during training: {e}")
            messagebox.showerror("Training Error", f"An error occurred during training: {str(e)}")
    
    def load_data(self):
        self.status_var.set(f"Status: Loading data for Subject {self.current_subject}, {self.current_task}...")
        self.root.update_idletasks()
        
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=5)
        
        progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        progress_bar.pack(fill=tk.X, expand=True)
        progress_bar.start()
        
        self.thread = threading.Thread(target=self.load_data_thread)
        self.thread.daemon = True
        self.thread.start()
        
        self.root.after(100, lambda: self.check_loading_thread(progress_frame, progress_bar))
    
    def check_loading_thread(self, progress_frame, progress_bar):
        if self.thread and self.thread.is_alive():
            self.root.after(100, lambda: self.check_loading_thread(progress_frame, progress_bar))
        else:
            progress_bar.stop()
            progress_frame.destroy()
            
            if self.data_loaded:
                self.prev_button.config(state='normal')
                self.playpause_button.config(state='normal')
                self.next_button.config(state='normal')
                self.train_button.config(state='normal')
                
                if self.auto_advance:
                    self.schedule_auto_advance()
            
            if not self.model_loaded:
                response = messagebox.askyesno(
                    "Train Model", 
                    "No model is currently loaded. Would you like to train a model with the loaded data?"
                )
                if response:
                    self.train_model()
    
    def load_data_thread(self):
        try:
            runs = task_runs.get(self.current_task, [4])
            
            raw_list = []
            
            for run in runs:
                success, file_path = download_physionet_data(self.current_subject, run)
                
                if success and file_path:
                    try:
                        self.status_var.set(f"Status: Loading file {file_path}...")
                        self.root.update_idletasks()
                        
                        raw = mne.io.read_raw_edf(file_path, preload=True)
                        
                        print(f"Loaded raw data: {raw.n_times} samples, {len(raw.ch_names)} channels")
                        if raw.n_times > 0 and len(raw.ch_names) > 0:
                            raw_list.append(raw)
                        else:
                            print(f"Warning: File {file_path} contains no data")
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        self.status_var.set(f"Error loading file {file_path}: {e}")
                        self.root.update_idletasks()
            
            if not raw_list:
                self.status_var.set(f"Error: No data loaded for Subject {self.current_subject}")
                self.root.update_idletasks()
                return
            
            raw_concat = mne.concatenate_raws(raw_list)
            print(f"Concatenated data: {raw_concat.n_times} samples")
            
            self.status_var.set("Status: Preprocessing EEG data...")
            self.root.update_idletasks()
            
            raw_concat.filter(l_freq=1.0, h_freq=45.0)
            raw_concat.notch_filter(freqs=[50, 60])
            raw_concat.set_eeg_reference('average', projection=False)
            
            events, event_id = mne.events_from_annotations(raw_concat)
            print(f"Found events: {event_id}")
            
            event_id_selected = {}
            
            for key, value in event_id.items():
                if 'T0' in key:
                    event_id_selected['rest'] = value
                elif 'T1' in key:
                    event_id_selected['left'] = value
                elif 'T2' in key:
                    event_id_selected['right'] = value
            
            if not event_id_selected:
                self.status_var.set(f"Error: No valid motor imagery events found in the data")
                self.root.update_idletasks()
                return
            
            print(f"Selected events: {event_id_selected}")
            
            self.status_var.set("Status: Creating epochs...")
            self.root.update_idletasks()
            
            epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                              tmin=0.5, tmax=3.5,
                              baseline=(0.5, 1.0),
                              preload=True)
            
            if len(epochs) == 0:
                self.status_var.set(f"Error: No valid epochs found in the data")
                self.root.update_idletasks()
                return
            
            print(f"Created {len(epochs)} epochs")
            
            self.raw_data = raw_concat
            self.epochs = epochs
            self.channel_names = epochs.ch_names
            self.time_axis = epochs.times
            self.fs = int(epochs.info['sfreq'])
            self.available_channels = epochs.ch_names
            
            self.selected_channels = []
            for ch in default_channels:
                matches = [name for name in self.available_channels if ch in name]
                if matches:
                    self.selected_channels.append(matches[0])
            
            while len(self.selected_channels) < 3 and len(self.available_channels) > len(self.selected_channels):
                for ch in self.available_channels:
                    if ch not in self.selected_channels:
                        self.selected_channels.append(ch)
                        break
                        
            self.selected_channels = self.selected_channels[:3]
            
            self.buffer_size = min(500, len(self.time_axis))
            self.signals_buffer = np.zeros((3, self.buffer_size))
            
            self.current_epoch = 0
            self.load_epoch_data()
            
            for i, (ax, _) in enumerate(self.eeg_axes):
                if i < len(self.selected_channels):
                    ax.set_title(self.selected_channels[i], fontsize=12)
            
            if self.epoch_data:
                self.update_state_display(self.epoch_data['state'])
            
            self.data_loaded = True
            self.epoch_var.set(f"Epoch: 1/{len(self.epochs)}")
            self.status_var.set(f"Status: Loaded data for Subject {self.current_subject}, {len(self.epochs)} epochs")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error loading data: {e}")
    
    def load_epoch_data(self):
        if not self.epochs or self.current_epoch >= len(self.epochs):
            return False
        
        try:
            ch_indices = []
            for ch_name in self.selected_channels:
                if ch_name in self.channel_names:
                    ch_indices.append(self.channel_names.index(ch_name))
            
            event_id = self.epochs.events[self.current_epoch, 2]
            
            event_dict = {v: k for k, v in self.epochs.event_id.items()}
            state = event_dict.get(event_id, 'rest')
            
            epoch_data = self.epochs[self.current_epoch].get_data()[0]
            
            if ch_indices:
                signals = epoch_data[ch_indices]
            else:
                signals = epoch_data[:3]
            
            while signals.shape[0] < 3:
                signals = np.vstack([signals, np.zeros_like(signals[0])])
            
            signal_range = signals.max() - signals.min()
            if signal_range < 0.1:
                scaling_factor = 10.0 / signal_range if signal_range > 0 else 10.0
                signals = signals * scaling_factor
                print(f"Applied auto-scaling factor: {scaling_factor:.2f}")
            
            self.feature_vector = extract_advanced_features(signals, self.fs)
            
            self.extract_epoch_features(signals, state)
            
            self.epoch_data = {
                'signals': signals,
                'time': self.time_axis,
                'state': state,
                'fs': self.fs
            }
            
            self.epoch_var.set(f"Epoch: {self.current_epoch + 1}/{len(self.epochs)}")
            
            if self.model_loaded and self.model is not None:
                self.classify_current_epoch()
            
            return True
            
        except Exception as e:
            print(f"Error loading epoch {self.current_epoch}: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error loading epoch {self.current_epoch}: {e}")
            return False
    
    def classify_current_epoch(self):
        if not self.model_loaded or self.model is None or not self.epoch_data:
            return
        
        try:
            if self.feature_vector is not None:
                features = self.feature_vector.reshape(1, -1)
                
                prediction = self.model.predict(features)[0]
                
                predicted_class = class_names[prediction % len(class_names)]
                
                if hasattr(self.model, 'predict_proba'):
                    probas = self.model.predict_proba(features)[0]
                    max_proba = max(probas)
                    confidence = f" (Confidence: {max_proba:.2f})"
                else:
                    confidence = ""
                
                self.status_var.set(f"Model prediction: {predicted_class.upper()}{confidence}")
                
        except Exception as e:
            print(f"Error classifying epoch: {e}")
            self.status_var.set(f"Error in classification: {e}")
    
    def extract_epoch_features(self, signals, state):
        features = []
        alpha_powers = []
        beta_powers = []
        theta_powers = []
        
        for i, signal in enumerate(signals):
            theta, alpha, beta, freqs, psd = self.extract_band_powers(signal)
            
            theta_powers.append(theta)
            alpha_powers.append(alpha)
            beta_powers.append(beta)
            
            features.append({
                'theta': theta,
                'alpha': alpha,
                'beta': beta,
                'channel': i,
                'freqs': freqs,
                'psd': psd
            })
        
        try:
            if signals.shape[1] > 0:
                f, t, Sxx = sg.spectrogram(
                    signals[0], 
                    fs=self.fs,
                    nperseg=min(256, signals.shape[1]),
                    noverlap=128,
                    scaling='spectrum'
                )
                
                f_mask = f <= 30
                spectrogram_data = {
                    'f': f[f_mask],
                    't': t,
                    'Sxx': Sxx[f_mask]
                }
            else:
                spectrogram_data = None
        except Exception as e:
            print(f"Error computing spectrogram: {e}")
            spectrogram_data = None
        
        self.features = features
        self.alpha_powers = alpha_powers
        self.beta_powers = beta_powers
        self.theta_powers = theta_powers
        self.spectrogram_data = spectrogram_data
    
    def extract_band_powers(self, signal, fs=None):
        if fs is None:
            fs = self.fs if self.fs else 160
            
        try:
            n_samples = len(signal)
            nperseg = min(256, n_samples)
            noverlap = nperseg // 2
            
            freqs, psd = sg.welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
            
            theta_band = (4, 8)
            alpha_band = (8, 13)
            beta_band = (13, 30)
            
            theta_idx = np.logical_and(freqs >= theta_band[0], freqs <= theta_band[1])
            alpha_idx = np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])
            beta_idx = np.logical_and(freqs >= beta_band[0], freqs <= beta_band[1])
            
            theta_power = np.mean(psd[theta_idx]) if np.any(theta_idx) else 0
            alpha_power = np.mean(psd[alpha_idx]) if np.any(alpha_idx) else 0
            beta_power = np.mean(psd[beta_idx]) if np.any(beta_idx) else 0
            
            return theta_power, alpha_power, beta_power, freqs, psd
            
        except Exception as e:
            print(f"Error in band power extraction: {e}")
            return 0, 0, 0, np.array([]), np.array([])
    
    def update_eeg_plot(self, frame):
        if not self.data_loaded or not self.epoch_data or self.paused:
            return []
        
        signals = self.epoch_data['signals']
        time = self.epoch_data['time']
        state = self.epoch_data['state']
        
        if frame % 30 == 0:
            min_val = signals.min()
            max_val = signals.max()
            print(f"Signal range: {min_val:.3f} to {max_val:.3f}")
            
            if max_val - min_val < 0.1:
                signals = signals * 10
                print("Auto-scaling signals to enhance visibility")
        
        n_samples = min(signals.shape[1], self.buffer_size)
        
        start_idx = int((frame % signals.shape[1]) * 0.8)
        end_idx = min(start_idx + n_samples, signals.shape[1])
        display_length = end_idx - start_idx
        
        visible_signals = signals[:, start_idx:end_idx]
        visible_time = time[start_idx:end_idx]
        
        if visible_signals.shape[1] < self.buffer_size:
            pad_width = self.buffer_size - visible_signals.shape[1]
            visible_signals = np.pad(visible_signals, ((0, 0), (0, pad_width)), mode='constant')
            
            if len(visible_time) < self.buffer_size:
                if len(visible_time) > 0:
                    dt = visible_time[1] - visible_time[0] if len(visible_time) > 1 else 1/self.fs
                    visible_time = np.concatenate([visible_time, 
                                                 visible_time[-1] + dt * np.arange(1, pad_width+1)])
                else:
                    visible_time = np.linspace(0, self.buffer_size/self.fs, self.buffer_size)
        
        for i, (ax, line) in enumerate(self.eeg_axes):
            if i < visible_signals.shape[0]:
                line.set_data(visible_time, visible_signals[i])
                
                ax.set_xlim(visible_time[0], visible_time[-1])
                
                y_min, y_max = visible_signals[i].min(), visible_signals[i].max()
                padding = max(0.5, (y_max - y_min) * 0.1)
                ax.set_ylim(y_min - padding, y_max + padding)
                
                if state == 'left' and i == 2:
                    self.erd_annotations[i].set_visible(True)
                    self.erd_annotations[i].set_text('ERD')
                    self.erd_annotations[i].set_bbox({
                        'boxstyle': 'round,pad=0.5',
                        'fc': 'white',
                        'ec': COLORS['states']['left'],
                        'alpha': 0.7
                    })
                elif state == 'right' and i == 0:
                    self.erd_annotations[i].set_visible(True)
                    self.erd_annotations[i].set_text('ERD')
                    self.erd_annotations[i].set_bbox({
                        'boxstyle': 'round,pad=0.5',
                        'fc': 'white',
                        'ec': COLORS['states']['right'],
                        'alpha': 0.7
                    })
                else:
                    self.erd_annotations[i].set_visible(False)
                    
                ylim = ax.get_ylim()
                height = ylim[1] - ylim[0]
                
                if i < len(self.highlight_boxes):
                    try:
                        self.highlight_boxes[i].remove()
                    except:
                        pass
                
                color = COLORS['states'][state]
                opacity = 0.2
                
                if (state == 'left' and i == 2) or (state == 'right' and i == 0):
                    opacity = 0.4
                
                self.highlight_boxes[i] = ax.add_patch(
                    Rectangle((visible_time[0], ylim[0]), 
                             visible_time[-1] - visible_time[0], height,
                             alpha=opacity, 
                             color=color,
                             zorder=0)
                )
        
        return [line for _, line in self.eeg_axes]
    
    def update_timefreq_plot(self, frame):
        if not self.data_loaded or not self.spectrogram_data or self.paused:
            return [self.spectrogram]
        
        f = self.spectrogram_data['f']
        t = self.spectrogram_data['t']
        Sxx = self.spectrogram_data['Sxx']
        
        if np.max(Sxx) > 0:
            Sxx_norm = Sxx / np.max(Sxx)
        else:
            Sxx_norm = Sxx
        
        self.spectrogram.set_array(Sxx_norm)
        
        if len(t) > 0:
            self.spectrogram.set_extent([t[0], t[-1], f[0], f[-1]])
            self.tf_ax.set_xlim(t[0], t[-1])
            self.tf_ax.set_ylim(f[0], f[-1])
        
        return [self.spectrogram]
    
    def update_feature_plots(self, frame):
        if not self.data_loaded or not self.features or self.paused:
            return []
        
        current_state = self.epoch_data['state']
        updated_artists = []
        
        radar_data = []
        for ch_idx, feature in enumerate(self.features):
            if ch_idx < 3:
                radar_data.extend([
                    feature['theta'],
                    feature['alpha'],
                    feature['beta']
                ])
        
        if radar_data:
            if max(radar_data) > 0:
                radar_data = [val / max(radar_data) * 10 for val in radar_data]
            else:
                radar_data = [0] * len(radar_data)
            
            radar_values = radar_data + [radar_data[0]]
            
            self.radar_line.set_ydata(radar_values)
            updated_artists.append(self.radar_line)
            
            if len(self.radar_ax.collections) > 0:
                self.radar_ax.collections[0].remove()
            self.radar_ax.fill(self.radar_angles, radar_values, 
                           color=COLORS['states'][current_state], alpha=0.3)
            
            max_val = max(radar_values)
            self.radar_ax.set_ylim(0, max(10.0, max_val * 1.1))
        
        for i, bar in enumerate(self.bars):
            if i < len(radar_data):
                bar.set_height(radar_data[i])
                updated_artists.append(bar)
        
        if radar_data:
            self.bar_ax.set_ylim(0, max(10.0, max(radar_data) * 1.1))
        
        self.ts_c3_data = np.roll(self.ts_c3_data, -1)
        self.ts_cz_data = np.roll(self.ts_cz_data, -1)
        self.ts_c4_data = np.roll(self.ts_c4_data, -1)
        
        if len(self.alpha_powers) >= 3:
            self.ts_c3_data[-1] = self.alpha_powers[0]
            self.ts_cz_data[-1] = self.alpha_powers[1]
            self.ts_c4_data[-1] = self.alpha_powers[2]
        
        self.ts_c3_line.set_ydata(self.ts_c3_data)
        self.ts_cz_line.set_ydata(self.ts_cz_data)
        self.ts_c4_line.set_ydata(self.ts_c4_data)
        updated_artists.extend([self.ts_c3_line, self.ts_cz_line, self.ts_c4_line])
        
        all_alpha_data = np.concatenate([self.ts_c3_data, self.ts_cz_data, self.ts_c4_data])
        if len(all_alpha_data) > 0:
            y_min, y_max = np.min(all_alpha_data), np.max(all_alpha_data)
            padding = max(0.1, (y_max - y_min) * 0.1)
            self.timeseries_ax.set_ylim(max(0, y_min - padding), y_max + padding)
        
        self.erd_signal = np.copy(self.baseline)
        erd_idx = np.logical_and(self.erd_x >= 2, self.erd_x <= 3)
        
        erd_depth = 0.4
        if len(self.alpha_powers) >= 3:
            if current_state == 'left' and self.alpha_powers[2] < 0.7 * self.alpha_powers[0]:
                erd_depth = 0.3
            elif current_state == 'right' and self.alpha_powers[0] < 0.7 * self.alpha_powers[2]:
                erd_depth = 0.3
        
        self.erd_signal[erd_idx] = self.baseline[erd_idx] * erd_depth
        
        self.erd_signal_line.set_ydata(self.erd_signal)
        updated_artists.append(self.erd_signal_line)
        
        self.erd_highlight.set_color(COLORS['states'][current_state])
        self.erd_signal_line.set_color(COLORS['states'][current_state])
        
        return updated_artists
    
    def update_topography(self, frame):
        if not self.data_loaded or not self.alpha_powers or not self.beta_powers or self.paused:
            return [self.alpha_heatmap, self.beta_heatmap]
        
        current_state = self.epoch_data['state']
        
        n = self.topo_grid_size
        
        positions = {
            'C3': (-0.5, 0),
            'Cz': (0, 0),
            'C4': (0.5, 0)
        }
        
        alpha_data = np.zeros((n, n))
        beta_data = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if self.topo_mask[i, j]:
                    point = (self.topo_x[j], self.topo_y[i])
                    
                    alpha_val = 0
                    beta_val = 0
                    total_weight = 0
                    
                    for ch_idx, (ch_name, pos) in enumerate(positions.items()):
                        dist = np.sqrt((point[0] - pos[0])**2 + (point[1] - pos[1])**2)
                        weight = 1 / (dist + 0.1)**2
                        
                        if ch_idx < len(self.alpha_powers):
                            alpha_val += weight * self.alpha_powers[ch_idx]
                            beta_val += weight * self.beta_powers[ch_idx]
                        
                        total_weight += weight
                    
                    if total_weight > 0:
                        alpha_data[i, j] = alpha_val / total_weight
                        beta_data[i, j] = beta_val / total_weight
        
        if np.max(alpha_data) > 0:
            alpha_data = alpha_data / np.max(alpha_data) * 2 - 1
        if np.max(beta_data) > 0:
            beta_data = beta_data / np.max(beta_data) * 2 - 1
        
        if current_state == 'left':
            for i in range(n):
                for j in range(n):
                    if self.topo_mask[i, j] and self.topo_x[j] > 0.2:
                        alpha_data[i, j] *= -0.5
                        beta_data[i, j] *= 1.5
        
        elif current_state == 'right':
            for i in range(n):
                for j in range(n):
                    if self.topo_mask[i, j] and self.topo_x[j] < -0.2:
                        alpha_data[i, j] *= -0.5
                        beta_data[i, j] *= 1.5
        
        self.alpha_heatmap.set_array(alpha_data)
        self.beta_heatmap.set_array(beta_data)
        
        return [self.alpha_heatmap, self.beta_heatmap]
    
    def update_state_display(self, state=None):
        if state is None and self.epoch_data:
            state = self.epoch_data['state']
        elif state is None:
            state = 'rest'
        
        self.current_state.set(state.upper())
        
        for s, (frame, label) in self.state_boxes.items():
            if s == state:
                label.configure(foreground='white')
                frame.configure(style='StateBox.TFrame')
                frame.configure(style='ActiveState.TFrame')
                
                if s == 'rest':
                    self.style.configure('ActiveState.TFrame', background=COLORS['states']['rest'])
                    self.state_value.configure(foreground=COLORS['states']['rest'])
                elif s == 'left':
                    self.style.configure('ActiveState.TFrame', background=COLORS['states']['left'])
                    self.state_value.configure(foreground=COLORS['states']['left'])
                elif s == 'right':
                    self.style.configure('ActiveState.TFrame', background=COLORS['states']['right'])
                    self.state_value.configure(foreground=COLORS['states']['right'])
            else:
                label.configure(foreground=COLORS['text'])
                frame.configure(style='StateBox.TFrame')
    
    def next_epoch(self):
        if self.data_loaded and self.epochs and self.current_epoch < len(self.epochs) - 1:
            self.current_epoch += 1
            self.load_epoch_data()
            if self.epoch_data:
                self.update_state_display(self.epoch_data['state'])
    
    def prev_epoch(self):
        if self.data_loaded and self.epochs and self.current_epoch > 0:
            self.current_epoch -= 1
            self.load_epoch_data()
            if self.epoch_data:
                self.update_state_display(self.epoch_data['state'])
    
    def toggle_play_pause(self):
        self.paused = not self.paused
        
        if self.paused:
            self.playpause_button.config(text="► Play")
            if self.auto_advance_id:
                self.root.after_cancel(self.auto_advance_id)
                self.auto_advance_id = None
        else:
            self.playpause_button.config(text="❚❚ Pause")
            if self.auto_advance:
                self.schedule_auto_advance()
    
    def toggle_auto_advance(self):
        self.auto_advance = self.auto_advance_var.get()
        
        if self.auto_advance and not self.paused:
            self.schedule_auto_advance()
        elif not self.auto_advance and self.auto_advance_id:
            self.root.after_cancel(self.auto_advance_id)
            self.auto_advance_id = None
    
    def schedule_auto_advance(self):
        if self.auto_advance_id:
            self.root.after_cancel(self.auto_advance_id)
        
        self.auto_advance_id = self.root.after(self.advance_interval, self.auto_advance_epoch)
    
    def auto_advance_epoch(self):
        if self.data_loaded and not self.paused and self.auto_advance:
            if self.current_epoch < len(self.epochs) - 1:
                self.next_epoch()
            else:
                self.current_epoch = 0
                self.load_epoch_data()
                if self.epoch_data:
                    self.update_state_display(self.epoch_data['state'])
            
            self.schedule_auto_advance()

if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicBCIVisualization(root)
    root.mainloop()