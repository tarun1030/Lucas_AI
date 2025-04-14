# Lucas AI - Voice Assistant Backend
# Enhanced with better models for coding, summarization, and reasoning

import os
import time
import datetime
import json
import logging
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import requests
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lucas AI Voice Assistant")

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    message: str
    task_type: Optional[str] = "general"  # general, coding, summarization, reasoning

# Model registry with specialized models for different tasks
MODEL_REGISTRY = {
    "general": {
        "small": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "medium": "facebook/opt-1.3b",
        "large": "facebook/opt-2.7b"
    },
    "coding": {
        "small": "codellama/CodeLlama-7b-hf",  # Smaller CodeLlama variant
        "medium": "Phind/Phind-CodeLlama-34B-v2",  # Good balance
        "large": "WizardLM/WizardCoder-Python-34B-V1.0"  # Better but requires more resources
    },
    "summarization": {
        "small": "facebook/bart-large-cnn",  # Lightweight summarization model
        "medium": "google/pegasus-xsum",  # Better quality summaries
        "large": "allenai/led-large-16384-arxiv"  # Long document summarization
    },
    "reasoning": {
        "small": "microsoft/phi-2",  # Good reasoning in small size
        "medium": "stabilityai/stablelm-zephyr-3b",  # Better reasoning capabilities
        "large": "google/flan-t5-xl"  # Advanced reasoning and instruction following
    }
}

# Custom agent classes for specialized capabilities
class AgentBase:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.load_model()
    
    def load_model(self):
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # If the tokenizer doesn't have a pad token, use eos token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine loading strategy based on device and model size
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                # Try 8-bit quantization for CPU
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                except ImportError:
                    # Fall back to standard loading
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
            
            # Set up generation pipeline
            self.generator = pipeline(
                     "text-generation",
                      model=self.model,
                      tokenizer=self.tokenizer,
                      max_new_tokens=200
                        )
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_response(self, prompt, max_tokens=200):
        try:
            response = self.generator(
                prompt,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            return generated_text[len(prompt):].strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error processing your request."

class CodingAgent(AgentBase):
    def process_query(self, query):
        # Specialized prompt for coding tasks
        prompt = f"""
        You are a coding assistant specialized in writing efficient, correct code.
        Please provide a solution for the following coding task.
        
        User Request: {query}
        
        Solution:
        ```
        """
        
        # Generate code with higher max tokens for code
        raw_response = self.generate_response(prompt, max_tokens=500)
        
        # Try to extract code between backticks if present
        if "```" in raw_response:
            parts = raw_response.split("```")
            if len(parts) >= 3:
                # Return just the code
                return parts[1].strip()
        
        return raw_response

class SummarizationAgent(AgentBase):
    def process_query(self, query):
        # Check if the query is a text to summarize or a request for summarization
        if len(query.split()) > 50:  # If long text, assume it's content to summarize
            text_to_summarize = query
        else:
            # It's likely a request to summarize something
            return "Please provide the text you'd like me to summarize."
        
        # For summarization models, use a different approach
        if "bart" in self.model_name or "pegasus" in self.model_name or "led" in self.model_name:
            summarizer = pipeline("summarization", model=self.model_name, device=0 if self.device == "cuda" else -1)
            try:
                # Truncate input if needed - check model's max length
                max_input_length = 1024  # Default fallback
                if hasattr(self.tokenizer, "model_max_length"):
                    max_input_length = min(self.tokenizer.model_max_length, 4096)
                
                encoded_input = self.tokenizer(text_to_summarize, truncation=True, max_length=max_input_length, return_tensors="pt")
                decoded_input = self.tokenizer.decode(encoded_input["input_ids"][0], skip_special_tokens=True)
                
                summary = summarizer(decoded_input, max_length=150, min_length=40, do_sample=False)
                return summary[0]['summary_text']
            except Exception as e:
                logger.error(f"Error in summarization: {e}")
                return "I encountered an error summarizing this text."
        else:
            # Use general approach for other models
            prompt = f"""
            Please summarize the following text concisely:
            
            {text_to_summarize}
            
            Summary:
            """
            return self.generate_response(prompt)

class ReasoningAgent(AgentBase):
    def process_query(self, query):
        # Specialized prompt for reasoning tasks
        prompt = f"""
        Please think through this problem step by step:
        
        {query}
        
        Step-by-step solution:
        1.
        """
        
        # Generate reasoning with higher max tokens and lower temperature for more precise thinking
        response = self.generator(
            prompt,
            do_sample=True,
            temperature=0.5,  # Lower temperature for more focused reasoning
            top_p=0.95,
            repetition_penalty=1.1,
            max_new_tokens=400,  # More tokens for detailed reasoning
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        reasoning = generated_text[len(prompt):].strip()
        
        # Clean up the response
        return reasoning

# Enhanced Lucas Agent with specialized agent capabilities
class LucasAgent:
    def __init__(self):
        self.conversation_history = []
        self.system_prompt = """
        You are Lucas AI, a helpful voice assistant. Always begin by greeting the user with "Good Morning/Afternoon/Evening" 
        based on their local time. Be concise, helpful, and friendly. If you don't know something, 
        admit it rather than making up information. Maintain a conversational, helpful tone at all times.
        """
        
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Size tier based on available resources
        if self.device == "cuda" and torch.cuda.get_device_properties(0).total_memory > 8e9:  # >8GB VRAM
            self.size_tier = "large"
        elif self.device == "cuda":  # <8GB VRAM
            self.size_tier = "medium"
        else:  # CPU
            self.size_tier = "small"
        
        logger.info(f"Using {self.size_tier} tier models based on available resources")
        
        # Initialize the general model first
        self.general_model_name = MODEL_REGISTRY["general"][self.size_tier]
        self.load_general_model()
        
        # Store specialized agents without initializing immediately to save memory
        self.specialized_agents = {}
        
    def load_general_model(self):
        logger.info(f"Loading general model: {self.general_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.general_model_name)
            
            # If the tokenizer doesn't have a pad token, use eos token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate optimizations
            if self.device == "cuda":
                # Use half precision for GPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.general_model_name, 
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                # For CPU, use 8-bit quantization if available
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.general_model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                except ImportError:
                    # If bitsandbytes not available, load normally
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.general_model_name,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
            
            # Create text generation pipeline
            self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200
        )
            logger.info(f"General model loaded successfully: {self.general_model_name}")
        except Exception as e:
            logger.error(f"Error loading general model: {e}")
            raise
    
    def get_specialized_agent(self, task_type):
        """Get or create a specialized agent for a specific task"""
        if task_type not in ["coding", "summarization", "reasoning"]:
            return None
        
        if task_type not in self.specialized_agents:
            # Create the specialized agent on demand
            model_name = MODEL_REGISTRY[task_type][self.size_tier]
            
            if task_type == "coding":
                self.specialized_agents[task_type] = CodingAgent(model_name, self.device)
            elif task_type == "summarization":
                self.specialized_agents[task_type] = SummarizationAgent(model_name, self.device)
            elif task_type == "reasoning":
                self.specialized_agents[task_type] = ReasoningAgent(model_name, self.device)
        
        return self.specialized_agents[task_type]
    
    def get_time_greeting(self):
        """Returns appropriate time-based greeting"""
        current_hour = datetime.datetime.now().hour
        
        if 5 <= current_hour < 12:
            return "Good Morning"
        elif 12 <= current_hour < 18:
            return "Good Afternoon"
        else:
            return "Good Evening"
    
    def format_prompt(self, user_input):
        """Format the prompt based on the model type"""
        if "TinyLlama" in self.general_model_name:
            # TinyLlama chat format
            prompt = "<|system|>\n"
            prompt += self.system_prompt.strip() + "\n"
            
            # Add conversation history (keep last 5 exchanges for context)
            for message in self.conversation_history[-10:]:
                if message["role"] == "user":
                    prompt += f"<|user|>\n{message['content']}\n"
                else:
                    prompt += f"<|assistant|>\n{message['content']}\n"
            
            # Add prompt for assistant to respond
            prompt += "<|assistant|>\n"
        
        elif "opt" in self.general_model_name.lower():
            # Format for OPT models
            prompt = f"System: {self.system_prompt}\n\n"
            
            # Add conversation history
            for message in self.conversation_history[-10:]:
                if message["role"] == "user":
                    prompt += f"Human: {message['content']}\n"
                else:
                    prompt += f"Assistant: {message['content']}\n"
            
            # Add prompt for assistant to respond
            prompt += "Assistant: "
        
        else:
            # Generic format for other models
            prompt = f"{self.system_prompt}\n\n"
            
            # Add conversation history
            for message in self.conversation_history[-10:]:
                if message["role"] == "user":
                    prompt += f"User: {message['content']}\n"
                else:
                    prompt += f"Lucas: {message['content']}\n"
            
            # Add prompt for assistant to respond
            prompt += "Lucas: "
        
        return prompt
    
    def clean_response(self, prompt, raw_response):
        """Clean up the model response based on model type"""
        assistant_response = raw_response[len(prompt):].strip()
        
        # Clean up response based on model type
        if "TinyLlama" in self.general_model_name:
            # For TinyLlama, stop at special tokens
            for stop_seq in ["<|user|>", "<|system|>"]:
                if stop_seq in assistant_response:
                    assistant_response = assistant_response.split(stop_seq)[0].strip()
        else:
            # For other models, stop at the first occurrence of markers
            for stop_seq in ["User:", "Human:", "Lucas:", "Assistant:"]:
                if stop_seq in assistant_response:
                    assistant_response = assistant_response.split(stop_seq)[0].strip()
        
        # Filter out any empty responses
        if not assistant_response.strip():
            assistant_response = "I'm here to help. What can I do for you?"
        
        return assistant_response
    
    def process_query(self, user_input, task_type="general"):
        """Process user query with specialized agent if appropriate"""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Detect task type if not specified
        if task_type == "general":
            # Simple keyword-based task detection
            lower_input = user_input.lower()
            if any(word in lower_input for word in ["code", "program", "function", "script", "algorithm"]):
                task_type = "coding"
            elif any(word in lower_input for word in ["summarize", "summary", "condense", "shorten"]):
                task_type = "summarization"
            elif any(word in lower_input for word in ["explain", "reason", "analyze", "solve", "step by step"]):
                task_type = "reasoning"
        
        logger.info(f"Processing query with task type: {task_type}")
        
        try:
            if task_type != "general":
                # Use specialized agent if available
                agent = self.get_specialized_agent(task_type)
                if agent:
                    response = agent.process_query(user_input)
                    # Save response to conversation history
                    self.conversation_history.append({"role": "assistant", "content": response})
                    return response
            
            # If no specialized agent or general query, use the general model
            prompt = self.format_prompt(user_input)
            
            # Generate response using the local model
            raw_response = self.generator(
                prompt,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                max_new_tokens=200,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text and clean it
            generated_text = raw_response[0]['generated_text']
            assistant_response = self.clean_response(prompt, generated_text)
            
            # Save clean response to conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I'm sorry, I encountered an error processing your request. Please try again."
    
    def handle_first_interaction(self):
        """Handle first interaction with appropriate greeting"""
        greeting = self.get_time_greeting()
        return f"{greeting} Sir, How can I help you?"

# Instance of Lucas Agent
lucas = LucasAgent()

@app.get("/")
async def root():
    return {"message": "Lucas AI Voice Assistant API", "version": "2.0"}

@app.get("/models")
async def get_models():
    """Get available models and their status"""
    model_info = {
        "device": lucas.device,
        "size_tier": lucas.size_tier,
        "general_model": lucas.general_model_name,
        "specialized_models": {
            task: MODEL_REGISTRY[task][lucas.size_tier] 
            for task in ["coding", "summarization", "reasoning"]
        },
        "loaded_agents": list(lucas.specialized_agents.keys())
    }
    return model_info

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Send initial greeting
    first_message = lucas.handle_first_interaction()
    await websocket.send_text(json.dumps({
        "sender": "Lucas", 
        "message": first_message,
        "task_type": "general"
    }))
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            task_type = message_data.get("task_type", "general")
            
            if user_message.lower() in ["exit", "quit", "goodbye", "bye"]:
                await websocket.send_text(json.dumps({
                    "sender": "Lucas",
                    "message": "Goodbye! Have a great day."
                }))
                break
            
            # Process user query with task type if specified
            response = lucas.process_query(user_message, task_type)
            
            # Send response back to client
            await websocket.send_text(json.dumps({
                "sender": "Lucas",
                "message": response,
                "task_type": task_type
            }))
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket connection: {e}")

# Enhanced REST API endpoint with task type specification
@app.post("/query")
async def query(request: QueryRequest):
    # If this is the first message, return greeting
    if not lucas.conversation_history:
        response = lucas.handle_first_interaction()
    else:
        response = lucas.process_query(request.message, request.task_type)
    
    return {
        "response": response,
        "task_type": request.task_type
    }

# Memory management endpoint
@app.post("/clear_memory")
async def clear_memory():
    """Clear conversation history and unload specialized models to free memory"""
    lucas.conversation_history = []
    
    # Unload specialized agents to free memory
    for task_type in list(lucas.specialized_agents.keys()):
        try:
            del lucas.specialized_agents[task_type]
        except Exception as e:
            logger.error(f"Error unloading {task_type} agent: {e}")
    
    # Run garbage collection
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"message": "Memory cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)