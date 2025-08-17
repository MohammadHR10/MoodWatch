# MoodWatch - Facial Action Units Reference

A comprehensive guide to understanding Facial Action Units (AUs) for emotion detection and analysis.

## 📸 Application Screenshots

### Dashboard Overview
![MoodWatch Dashboard](images/dashboard-overview.png)
*Main Streamlit dashboard showing emotion analysis charts and real-time data*

### Camera Preview
![Camera Preview](images/camera-preview.png) 
*Live camera feed with emotion detection overlay*

### LLM Interpretation
![LLM Interpretation](images/llm-interpretation.png)
*AI-powered emotion interpretation and insights*

---

## What are Action Units (AUs)?

Action Units are the fundamental components of the Facial Action Coding System (FACS), developed by Paul Ekman and Wallace Friesen. They represent the basic movements of facial muscles that combine to create facial expressions and emotional displays.

## Action Units Reference Table

| AU       | Common Name / Primary Muscle(s)                           | What You Actually See                       | Typical Emotion Hints\*           | Practical Notes                                                       |
| -------- | --------------------------------------------------------- | ------------------------------------------- | --------------------------------- | --------------------------------------------------------------------- |
| **AU01** | Inner brow raiser (frontalis, pars medialis)              | Inner brows lift, brows "knit up"           | Sadness, fear, concern            | Often pairs with AU04 in sadness; sensitive to lighting/shadows.      |
| **AU02** | Outer brow raiser (frontalis, pars lateralis)             | Outer brows lift                            | Surprise, fear                    | With AU01+AU05 gives strong surprise/fear cues.                       |
| **AU04** | Brow lowerer (corrugator, depressor supercilii, procerus) | Frown/vertical glabella lines               | Anger, sadness, effort            | Your **furrow** proxy. Counteracted by smiles (AU12).                 |
| **AU06** | Cheek raiser (orbicularis oculi)                          | Cheek lift/crow's feet, eye "smile"         | Joy (Duchenne when with AU12)     | Key for genuine smiles: **AU6 + AU12**.                               |
| **AU07** | Lid tightener (orbicularis oculi)                         | Eyes narrow/tense                           | Anger, effort, pain               | Helps disambiguate anger vs neutral.                                  |
| **AU09** | Nose wrinkler (levator labii superioris alaeque nasi)     | Nose wrinkles, nostrils flare               | Disgust                           | Often brief; combine with AU10/17.                                    |
| **AU10** | Upper lip raiser (levator labii)                          | Upper lip lifts, teeth may show             | Disgust, contempt                 | With AU09 strengthens disgust.                                        |
| **AU12** | Lip corner puller (zygomaticus major)                     | Smile—corners pull up                       | Joy/positive valence              | Your **smile** proxy and main positive valence driver.                |
| **AU14** | Dimpler (buccinator)                                      | Lip corners pull **laterally** (flat smirk) | Contempt (esp. unilateral)        | Look for one-sided activation. Subtle.                                |
| **AU15** | Lip corner depressor (depressor anguli oris)              | Corners pull **down**                       | Sadness                           | With AU01/AU04 → strong sadness pattern.                              |
| **AU17** | Chin raiser (mentalis)                                    | Chin puckers, lower lip protrudes           | Disgust, sadness, doubt           | Modulates mouth shape in disgust.                                     |
| **AU20** | Lip stretcher (risorius)                                  | Corners pull **sideways** (teeth bared)     | Fear grimace, tension             | Common in fear/surprise blends.                                       |
| **AU23** | Lip tightener (orbicularis oris)                          | Lips pressed tight                          | Anger, determination              | Helps you score "anger_tense".                                        |
| **AU25** | Lips part                                                 | Lips slightly parted                        | Surprise, speech, arousal         | Weak mouth opening; combine with AU26.                                |
| **AU26** | Jaw drop                                                  | Mouth clearly open                          | Surprise, fear, speaking, arousal | Your **mouth_open** proxy. Stronger than AU25.                        |
| **AU45** | Blink (coded as \*\_c presence)                           | Eye blink/rapid closures                    | Fatigue, stress, dryness          | Use **rate** per minute; don't treat single blinks as arousal spikes. |

## Key Emotion Patterns

### 🙂 **Joy/Happiness**

- **Primary**: AU12 (lip corner puller) + AU06 (cheek raiser)
- **Genuine Smile (Duchenne)**: AU12 + AU06 together
- **Note**: AU06 is crucial for distinguishing genuine from fake smiles

### 😢 **Sadness**

- **Primary**: AU01 (inner brow raiser) + AU04 (brow lowerer) + AU15 (lip corner depressor)
- **Pattern**: Inner brows lift while frowning, mouth corners pull down
- **Note**: AU01+AU04 combination is a strong sadness indicator

### 😠 **Anger**

- **Primary**: AU04 (brow lowerer) + AU07 (lid tightener) + AU23 (lip tightener)
- **Pattern**: Furrowed brow, narrowed eyes, pressed lips
- **Note**: AU04 is counteracted by smiles (AU12)

### 😨 **Fear**

- **Primary**: AU01 (inner brow raiser) + AU02 (outer brow raiser) + AU20 (lip stretcher) + AU26 (jaw drop)
- **Pattern**: Raised brows, wide eyes, mouth open, lips stretched
- **Note**: AU01+AU02+AU05 gives strong fear cues

### 😮 **Surprise**

- **Primary**: AU01 + AU02 (brow raisers) + AU25/AU26 (lips part/jaw drop)
- **Pattern**: Raised brows, wide eyes, open mouth
- **Note**: Often brief and transitional

### 🤢 **Disgust**

- **Primary**: AU09 (nose wrinkler) + AU10 (upper lip raiser) + AU17 (chin raiser)
- **Pattern**: Wrinkled nose, raised upper lip, puckered chin
- **Note**: Often brief; combinations strengthen the signal

### 😏 **Contempt**

- **Primary**: AU14 (dimpler) - especially unilateral
- **Pattern**: One-sided lip corner pull (flat smirk)
- **Note**: Subtle and often asymmetrical

## 🚀 How to Run the MoodWatch Application

### Prerequisites & Installation

**Before running the application, make sure you have:**

1. **Python 3.8+** installed
2. **OpenFace** toolkit installed and configured
3. **Webcam** access permissions

### Local Installation Steps

1. **Clone and navigate to the project:**

   ```bash
   git clone <repository-url>
   cd MoodWatch
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Clear previous data (for fresh AU detection):**

   ```bash
   # Option 1: Remove all files inside processed folder (keeps the folder)
   rm -rf processed/*

   # Option 2: Remove entire processed folder (it will be recreated automatically)
   rm -rf processed/
   ```

   **Note**: This clears any previous emotion detection data for a fresh start.

4. **Set up environment variables:**
   Create a `.env` file with your OpenFace and API configurations:
   ```bash
   OPENFACE_BIN=/path/to/FeatureExtraction
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Quick Start

1. **Navigate to the MoodWatch directory:**

   ```bash
   cd MoodWatch
   ```

2. **Start the camera scheduler (Terminal 1):**

   ```bash
   python -m app.camera_schedule
   ```

3. **Launch the Streamlit web interface (Terminal 2):**
   ```bash
   ./venv/bin/streamlit run app/streamlit_app.py
   ```

### What Each Component Does

- **Camera Scheduler**: Manages camera capture and emotion detection processing
- **Streamlit App**: Provides the web-based dashboard for viewing results and analytics

### Accessing the Application

After running both commands, open your web browser and navigate to the URL shown in Terminal 2 (typically `http://localhost:8501`) to access the MoodWatch dashboard.

## Implementation Notes

### Key Proxies for Coding

- **Smile**: AU12 (lip corner puller)
- **Furrow**: AU04 (brow lowerer)
- **Mouth Open**: AU26 (jaw drop)
- **Genuine Smile**: AU12 + AU06 combination

### Technical Considerations

- **Lighting Sensitivity**: AU01 and AU04 are particularly sensitive to shadows
- **Blink Rate**: Use AU45 frequency per minute, not individual occurrences
- **Unilateral vs Bilateral**: Pay attention to asymmetrical activations (especially AU14)
- **Combination Effects**: Emotions are typically combinations of multiple AUs
- **Duration**: Some AUs (like AU09 for disgust) are typically brief

## Getting Started

1. **Basic Detection**: Start with the key proxies (AU12, AU04, AU26)
2. **Emotion Classification**: Use the combination patterns above
3. **Validation**: Cross-reference multiple AUs for stronger confidence
4. **Temporal Analysis**: Consider duration and timing of AU activations

## References

- Facial Action Coding System (FACS) by Paul Ekman and Wallace Friesen
- OpenFace toolkit for automated facial behavior analysis
- Emotion recognition research and applications

---

_Note: Emotion hints are guidelines, not absolute rules. Context, culture, and individual differences all play important roles in emotion recognition._
