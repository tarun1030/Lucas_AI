# Lucas AI Voice Assistant

Lucas AI is an advanced neural voice assistant with specialized AI models for different tasks. It provides a responsive interface with both voice and text input capabilities, making it an intuitive AI assistant for various needs.

## Features

- **Voice Activation**: Wake word "Lucas" for hands-free operation
- **Multi-modal Interaction**: Support for both voice and text input
- **Task-specific AI Models**: Specialized models for:
  - General conversation
  - Code generation
  - Text summarization
  - Complex reasoning
- **Responsive UI**: High-quality futuristic interface with real-time status indicators
- **Memory Management**: Ability to clear conversation history to free resources
- **WebSocket Communication**: Real-time bidirectional server-client communication
- **Resource-adaptive**: Automatically selects model size based on available hardware

## Technical Architecture

### Backend (FastAPI)

The backend system uses FastAPI with WebSockets to provide real-time communication with the frontend. It includes:

- Model registry with specialized AI models for different tasks
- Resource-aware model loading based on available hardware (CPU/GPU)
- Memory optimization with on-demand loading of specialized models
- Custom agent classes for different types of tasks

### Frontend

The frontend is built with HTML, CSS, and JavaScript, featuring:

- Futuristic UI with animated elements
- WebSocket connection for real-time communication
- Speech recognition for voice commands
- Text-to-speech for spoken responses
- Task type selection for different AI capabilities
- System information display

## Hardware Requirements

The system is adaptive to available resources:

- **Large tier**: CUDA-capable GPU with >8GB VRAM
- **Medium tier**: CUDA-capable GPU with <8GB VRAM
- **Small tier**: CPU-only operation (with quantization when possible)

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js and npm (for serving the frontend)
- PyTorch with CUDA support (for GPU acceleration)
- Modern web browser with WebSpeech API support

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/lucas-ai-voice-assistant.git
   cd lucas-ai-voice-assistant
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```
   cd frontend
   npm install
   ```

### Running the Application

1. Start the backend server:
   ```
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. Serve the frontend:
   ```
   cd frontend
   npm start  # or python -m http.server 3000
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## API Endpoints

- `/` - Root endpoint with API info
- `/models` - Get information about available models
- `/ws` - WebSocket endpoint for real-time communication
- `/query` - REST API endpoint for text queries
- `/clear_memory` - Endpoint to clear conversation history and unload models

## Model Registry

The system uses different models based on the task and available resources:

### General Conversation
- Small: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Medium: facebook/opt-1.3b
- Large: facebook/opt-2.7b

### Coding
- Small: codellama/CodeLlama-7b-hf
- Medium: Phind/Phind-CodeLlama-34B-v2
- Large: WizardLM/WizardCoder-Python-34B-V1.0

### Summarization
- Small: facebook/bart-large-cnn
- Medium: google/pegasus-xsum
- Large: allenai/led-large-16384-arxiv

### Reasoning
- Small: microsoft/phi-2
- Medium: stabilityai/stablelm-zephyr-3b
- Large: google/flan-t5-xl

## Usage

1. Say "Lucas" to activate voice recognition
2. Speak your request or type it in the input field
3. For specialized tasks, select the appropriate task type from the dropdown
4. View the response in the chat window and hear the spoken reply


## Acknowledgements

- Hugging Face for providing pre-trained models
- FastAPI for the backend framework
- WebSpeech API for voice recognition capabilities
