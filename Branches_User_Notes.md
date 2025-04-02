# User Notes (ml-integration-improvments)

## Application Overview

-   **Two Main Applications**:
    1.  `ecg_viewer.py`: The original application.
    2.  `standalone_ecg_simulator.py`: A newer application that can generate ECG signals in real-time. This simulator allows control over noise levels and heart rate.

## Machine Learning Integration

-   **Goal**: Integrate an ML model (`ML Model 2.0`) to classify ECG signals within the application.
-   **Current Status**:
    -   The application attempts to load the `ML Model 2.0`.
    -   If loading fails, it falls back to a "demo model", which is rule-based.
-   **Models**:
    -   **Original Model**: An earlier version.
    -   **ML Model 2.0**: A newer, better model trained on a larger dataset. This is the target model for integration.
    -   **Demo Model**: A rule-based classifier used as a fallback.

## Current Issues

-   **Incorrect Heart Rate Display**:
    -   In the `standalone_ecg_simulator.py`, when generating an ECG signal (e.g., at 40 BPM), the real-time plot shows the correct heart rate.
    -   However, the "Additional Metrics" section at the bottom displays an incorrect heart rate (e.g., showing 154 BPM when set to 40 BPM). This issue seems related to the *demo model's* calculations.
-   **ML Model Integration Failure**: The `ML Model 2.0` is not successfully implemented yet; the application currently relies on the demo fallback.

## Next Steps

1.  **Fix Demo Model Heart Rate Calculation**: Correct the heart rate display in the "Additional Metrics" section to accurately reflect the generated signal's heart rate. This is crucial to verify the demo model's functionality.
2.  **Verify/Fix Demo Model Logic**: After fixing the metrics display, confirm the *classification* logic of the demo model works correctly. If not, fix it.
3.  **Fix ML Model 2.0 Integration**: Once the demo model is reliable, attempt to fix the integration issues with `ML Model 2.0` so it can be used instead of the fallback. 