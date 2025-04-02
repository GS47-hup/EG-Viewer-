# ECG Viewer Project Goals

This document outlines the primary objectives for the ECG Viewer application development.

## Core Objectives:

1.  **Realistic ECG Generator:**
    *   Create and refine the `ECGGenerator` (`ecg_simulator.py`) to produce ECG waveforms highly representative of signals obtained from a real person using an AD8232 sensor connected via Arduino.
    *   The generator must be capable of simulating both **Normal Sinus Rhythm** and various common **Abnormal Rhythms** (e.g., Tachycardia, Bradycardia, potentially others like Atrial Fibrillation if feasible later).
    *   Realism in the generated signal is crucial for accurately testing the rhythm classification feature.

2.  **ECG Rhythm Classifier:**
    *   Implement a feature within the ECG Viewer application (`ecg_viewer.py`) that analyzes the incoming ECG data, whether it originates from the actual Arduino/Serial port OR the internal `ECGGenerator`.
    *   This feature must classify the underlying heart rhythm as **"Normal"** or **"Abnormal"**.
    *   Classification logic will likely involve analyzing parameters such as:
        *   Heart Rate (BPM)
        *   Regularity of RR intervals (Heart Rate Variability)
        *   Potentially QRS complex morphology or other waveform features.
        *   This might utilize the existing `MLClassifierUI` framework or start with simpler rule-based algorithms.
    *   Clearly display the classification result ("Normal" / "Abnormal") within the application's user interface.

3.  **(Future Goal) Arduino Integration:**
    *   Once the ECG Generator and Rhythm Classifier are functioning reliably within the software:
        *   Extend the application to transmit the "Normal" / "Abnormal" status signal back to the connected Arduino device via the serial connection.
        *   The separate Arduino sketch (firmware) will then interpret this signal to control corresponding LEDs:
            *   Green LED indicates a **Normal** classified rhythm.
            *   Red LED indicates an **Abnormal** classified rhythm.

## Current Status (on `generator-fixes` branch):

*   Basic `ECGGenerator` implemented in `ecg_simulator.py`.
*   Generator integrated into `ecg_viewer.py`.
*   Peak detection and BPM calculation adapted in `_ecg_math.py` to work with the generator's output.
*   UI elements for ML classification exist (`ml_classifier_ui.py`) but are not yet performing actual rhythm classification.

## Next Steps:

*   Address any remaining issues with the current generator's realism and stability.
*   Define specific criteria/rules or develop/integrate a model for classifying rhythms as "Normal" vs. "Abnormal".
*   Implement the classification logic within the application.
*   Update the UI to display the classification result.
*   Refine the generator to produce distinct and recognizable abnormal ECG patterns for testing. 