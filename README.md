# 🧠 NEUROSENSE – Brain-Controlled Prosthetic System

NEUROSENSE is an innovative Brain-Computer Interface (BCI) system that empowers individuals with limb loss to control a prosthetic arm using their thoughts alone. At the core of this breakthrough is a high-resolution 64-channel EEG dataset, processed on powerful cloud platforms for unmatched speed, precision, and scalability.

Unlike conventional systems dependent on limited hardware like Raspberry Pi or Jetson Nano, NEUROSENSE offloads computation to the cloud—enabling real-time, high-accuracy prosthetic control that’s both affordable and globally accessible.

🧬 Why NEUROSENSE Matters
🔎 According to the WHO, over 3 million people in India live with limb amputations.
NEUROSENSE by Team Zeka-Hackprix is our stride toward restoring natural, intuitive mobility—one thought at a time.
🚀 Key Innovations
🌐 Cloud-Native Architecture – AWS / Azure / Google Cloud ensure lightning-fast inference with global availability
🧠 64-Channel EEG Input – Captures complex brainwave patterns with fine spatial resolution
🧠⚙️ Smart Signal Interpretation – Advanced ensemble models (SVM, LDA, Gradient Boosting + Stacking) for robust motor imagery decoding
🎮 Live Prosthetic Simulation – Real-time virtual arm that mirrors thought-induced intent
📊 EEG Visualization Suite – Dynamic dashboards for brain signal monitoring and topographic mapping
💰 Cost-Effective & Scalable – Eliminates dependence on expensive edge hardware

🏗️ System Architecture
EEG Acquisition 
     ↓
Preprocessing & Noise Filtering
     ↓
Feature Extraction (Time, Frequency, CSP)
     ↓
Ensemble Machine Learning (SVM + LDA + GBoost)
     ↓
Cloud-Based Inference (AWS/Azure/GCP)
     ↓
Prosthetic Arm Control (Simulation/Real)


📁 Project Structure
File	Description
train2.py	Trains ensemble ML models on processed EEG features
test2.py	Evaluates classification accuracy, latency, and model robustness
simu_or_real.py	Interface switcher for simulated vs real-time EEG data
dynamic_bci_visualization.py	Live heatmaps and topographic maps of EEG signals
bci_arm_simulation.py	Virtual prosthetic arm controlled via decoded brain signals
eeg_visualisation.py	Multi-channel EEG waveform dashboard with live plots and signal analysis
🔮 What Sets NEUROSENSE Apart
🌍 Global-first: Easily deployable across borders with cloud-based access
⏱️ Zero Lag: Fast model response using high-performance compute instances
🧠 Medical-Grade Focus: Ready for integration with clinical-grade EEG systems
🎓 Research + Real-world Fusion: Bridges academia, healthcare, and industry
🤖 Expandable Framework: Ready for integration with robotics, AR, and haptics
🌟 Vision Ahead
Imagine a world where mobility is powered not by limbs, but by neural intent. Where a prosthetic doesn’t feel external, but like an extension of the mind.
NEUROSENSE isn’t just a project—it’s a paradigm shift in neurotechnology.

