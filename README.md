# Age and Emotion Detection from Male Voice

This project implements a machine learning pipeline to analyze voice recordings. It performs the following tasks:

1.  **Gender Detection:** Identifies the gender of the speaker.
2.  **Input Filtering:** Accepts only **male** voices for further processing. Female voices are rejected.
3.  **Age Group Estimation:** Predicts the age group (e.g., Youth, Adult, Senior) for accepted male voices.
4.  **Conditional Emotion Detection:** If the predicted age group for a male voice is "Senior" (defined as > 60 years old), the system also predicts the speaker's emotion (e.g., Happy, Sad, Angry, Neutral).

The project includes scripts for training the necessary models and a graphical user interface (GUI) built with Tkinter for easy interaction.

## Features

*   Loads audio files (`.wav`, `.mp3`).
*   Extracts MFCC features using `librosa`.
*   Trains separate models (using `scikit-learn`) for Gender, Age Group, and Emotion classification.
*   Provides evaluation metrics (Accuracy, Precision, Recall, Confusion Matrix) during training.
*   Requires models to achieve at least 70% accuracy on the test set (checked during training).
*   Saves trained models and scalers using `joblib`.
*   A user-friendly GUI (`tkinter`) to:
    *   Upload audio files.
    *   Display processing status and results based on the defined logic.
    *   Reject non-male voices explicitly.

## Technology Stack

*   **Python 3.x**
*   **Librosa:** For audio feature extraction.
*   **Scikit-learn:** For building and evaluating machine learning models (SVC, RandomForestClassifier, etc.).
*   **NumPy:** For numerical operations.
*   **Joblib:** For saving and loading trained models.
*   **Tkinter:** For the graphical user interface.

## Setup and Installation

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment:
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install System Dependencies (If needed):**
    *   **Tkinter:** Usually included with Python. If you encounter errors like `ModuleNotFoundError: No module named '_tkinter'`, install it via your system's package manager:
        *   Debian/Ubuntu: `sudo apt-get update && sudo apt-get install python3-tk`
        *   Fedora: `sudo dnf install python3-tkinter`
        *   macOS: Usually included. If using Homebrew Python, it should be there.
    *   **FFmpeg:** Required by `librosa` to load many common audio formats (especially MP3).
        *   Debian/Ubuntu: `sudo apt-get install ffmpeg`
        *   Fedora: `sudo dnf install ffmpeg`
        *   macOS (using Homebrew): `brew install ffmpeg`
        *   Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your system's PATH.

## Dataset Preparation (CRITICAL STEP)

*   **You MUST provide your own datasets.** The training script (`train_models.py`) currently uses **simulated data** as a placeholder.
*   You need separate labeled audio datasets for:
    *   **Gender:** Audio files labeled as 'male' or 'female'.
    *   **Age Group:** Audio files (preferably only male voices, as per project requirements) labeled with age groups (e.g., 'youth', 'adult', 'senior'). Define your age ranges clearly.
    *   **Emotion:** Audio files (preferably only *senior* male voices, or at least *male* voices) labeled with emotions (e.g., 'happy', 'sad', 'neutral', 'angry').
*   **Modify `train_models.py`:**
    *   Locate the `load_simulated_data` function.
    *   **Replace** the simulation logic with your actual data loading code. This code should:
        *   Find your audio files.
        *   Read their corresponding labels (gender, age group, emotion).
        *   Use the `extract_features` function to process each audio file.
        *   Return two NumPy arrays: one for features and one for numerical labels (encoded according to the `*_map_train` dictionaries in the script).
    *   Ensure the label mappings (`gender_map_train`, `age_map_train`, `emotion_map_train`) in `train_models.py` match how you numerically encode your labels.
    *   Adjust feature extraction parameters (`N_MFCC`, `MAX_PAD_LEN`) if necessary, based on analysis of your data.

## Training the Models

1.  **Prepare your data** and modify `train_models.py` as described above.
2.  Run the training script from your terminal:
    ```bash
    python train_models.py
    ```
3.  The script will:
    *   Load data for each task (Gender, Age, Emotion).
    *   Train the respective models.
    *   Print evaluation metrics (Accuracy, Classification Report, Confusion Matrix) to the console.
    *   **Check if accuracy meets the 70% minimum threshold.** Warnings will be printed if it doesn't.
    *   Save the trained models (`.joblib`) and scalers (`.joblib`) into the `saved_models/` directory.

## Running the Application

1.  Ensure the `train_models.py` script has run successfully and the `saved_models/` directory exists and contains the `.joblib` files.
2.  Run the GUI application script:
    ```bash
    python gui_app.py
    ```
3.  The application window will appear.
4.  Click the "Upload Audio File" button and select a `.wav` or `.mp3` file.
5.  The application will process the audio and display the results according to the project logic:
    *   If female voice detected -> "Upload male voice."
    *   If male voice (<= 60) -> Display predicted Age Group.
    *   If male voice (> 60 / Senior) -> Display predicted Age Group, Status: Senior Citizen, and predicted Emotion.

## Project Structure
