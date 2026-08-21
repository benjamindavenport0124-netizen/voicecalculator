# Whisper Voice Calculator

A desktop calculator that turns spoken mathematics into symbolic expressions and evaluates them. The application records microphone input, transcribes it locally with OpenAI Whisper, converts spoken math into a SymPy-compatible expression, and displays the result in a Tkinter interface.

The project is intentionally kept as a single Python file. It grew as an exploration of speech recognition, natural-language parsing, symbolic mathematics, and numerical problem solving.

## Features

- Voice input through the system microphone
- Local speech-to-text transcription with Whisper
- A simple Tkinter desktop interface
- Basic arithmetic, percentages, powers, roots, and fractions
- Trigonometric and inverse-trigonometric functions
- Logarithms, constants, and imaginary numbers
- Symbolic simplification and equation solving
- Systems of equations with symbolic and numerical fallbacks
- Derivatives, integrals, limits, and Taylor/Maclaurin series
- Radian and degree modes

## How it works

```text
Microphone
    -> temporary WAV recording
    -> Whisper transcription
    -> spoken-math parser
    -> SymPy/SciPy evaluation
    -> result in the Tkinter window
```

## Requirements

- Python 3.10 or 3.11 recommended
- A working microphone
- Tkinter (normally included with Python on Windows and macOS)
- [FFmpeg](https://ffmpeg.org/) available on your system `PATH`
- Internet access on the first run so Whisper can download its model
- Several gigabytes of free storage and memory for Whisper's `medium` model

A CUDA-capable GPU is optional. Whisper can run on a CPU, but transcription with the `medium` model may be slow.

### Platform notes

- **Windows:** install a standard Python distribution from python.org so Tkinter is included. Install FFmpeg separately and add it to `PATH`.
- **macOS:** FFmpeg can be installed with `brew install ffmpeg`. You may need to grant microphone permission to Terminal or your Python launcher.
- **Linux:** install FFmpeg, Tkinter, and PortAudio using your distribution's package manager. Package names commonly include `ffmpeg`, `python3-tk`, and `portaudio19-dev`.

## Installation

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/benjamindavenport0124-netizen/voice-calculator.git
   cd voice-calculator
   ```

2. Create and activate a virtual environment:

   **Windows (PowerShell)**

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the Python dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Confirm that FFmpeg is installed:

   ```bash
   ffmpeg -version
   ```

## Usage

Run the application:

```bash
python voice_calculator.py
```

You can type an expression and select **Calculate**, or select **Start Recording**, speak a problem, and then select **Stop Recording**. The first launch can take longer because Whisper downloads and loads the `medium` model.

Example spoken requests include:

- “two plus three times four”
- “square root of sixteen”
- “sine of pi over two”
- “solve x squared minus five x plus six equals zero”
- “find the derivative of x cubed”

Speech parsing is heuristic, so phrasing and transcription accuracy can affect the result.

## Known limitations

- The Whisper `medium` model has substantial download, memory, and startup costs.
- Transcription runs synchronously after recording, so the interface may appear unresponsive until it finishes.
- Each voice request creates a `temp_<timestamp>.wav` file in the working directory. These recordings are ignored by Git but are not deleted automatically.
- Stopping immediately, before audio frames have been captured, can cause an error.
- Numerical solvers may return approximate results or fail on difficult equations.
- The expression parser was designed for a local desktop app. Do not expose it as a public web service without validating and sandboxing input.

## Privacy

Audio is recorded to a temporary local WAV file and transcribed by a locally running Whisper model. The application does not intentionally upload recordings to an external transcription API. Delete generated `temp_*.wav` files when they are no longer needed.

## Project structure

```text
voice-calculator/
├── LICENSE
├── voice_calculator.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Future improvements

- Delete temporary recordings automatically after transcription
- Load the model in the background and show startup progress
- Move transcription off the GUI thread
- Let users choose a smaller or larger Whisper model
- Add automated tests for spoken-math conversions
- Separate the interface, parser, recording, and math engine into modules

## License

This project is available under the [MIT License](LICENSE).
