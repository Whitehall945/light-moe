"""
Gradio Demo for Light-MoE

Interactive web interface for MoE model inference with:
- Real-time text generation
- Expert activation visualization
- Performance metrics display
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Generator

try:
    import gradio as gr
except ImportError:
    gr = None
    print("Gradio not installed. Run: pip install gradio")

import torch


def create_demo(model_path: Optional[str] = None):
    """Create Gradio demo interface."""
    
    if gr is None:
        raise ImportError("Please install gradio: pip install gradio")
    
    # Mock model for demo (replace with real Light-MoE engine)
    class MockEngine:
        def __init__(self):
            self.expert_activations = []
            
        def generate(self, prompt: str, max_tokens: int = 128, 
                    temperature: float = 0.7, top_p: float = 0.9) -> str:
            # Simulate generation
            time.sleep(0.1)
            return prompt + " [Generated text would appear here with a loaded model]"
        
        def generate_stream(self, prompt: str, max_tokens: int = 128,
                           temperature: float = 0.7, top_p: float = 0.9) -> Generator:
            words = ["This", "is", "a", "streaming", "demo", "of", "Light-MoE", 
                    "inference", "engine", ".", "The", "actual", "model", 
                    "would", "generate", "real", "text", "."]
            
            for i, word in enumerate(words[:max_tokens]):
                time.sleep(0.05)
                self.expert_activations.append({
                    "token": word,
                    "expert_1": (i % 8),
                    "expert_2": ((i + 3) % 8),
                    "weight_1": 0.7,
                    "weight_2": 0.3,
                })
                yield word + " "
        
        def get_expert_stats(self) -> dict:
            if not self.expert_activations:
                return {"message": "No generations yet"}
            
            expert_counts = [0] * 8
            for act in self.expert_activations[-50:]:  # Last 50 tokens
                expert_counts[act["expert_1"]] += 1
                expert_counts[act["expert_2"]] += 1
            
            return {
                "total_tokens": len(self.expert_activations),
                "expert_distribution": expert_counts,
            }
    
    # Initialize engine
    try:
        from light_moe import LightMoEEngine
        if model_path:
            engine = LightMoEEngine(model_path=model_path)
        else:
            print("No model path provided, using mock engine")
            engine = MockEngine()
    except ImportError:
        print("Light-MoE not available, using mock engine")
        engine = MockEngine()
    
    # Define interface
    with gr.Blocks(title="Light-MoE Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 Light-MoE Inference Demo
        
        High-performance Mixture-of-Experts inference engine with CuTe-optimized kernels.
        
        **Features:**
        - Expert Parallelism across multiple GPUs
        - Fused Gate + TopK kernels
        - Grouped GEMM with Tensor Core acceleration
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                    value="The future of artificial intelligence is",
                )
                
                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=16, maximum=512, value=128, step=16,
                        label="Max Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                        label="Top-P"
                    )
                
                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")
                
                # Output section
                output = gr.Textbox(
                    label="Generated Text",
                    lines=10,
                    interactive=False,
                )
                
            with gr.Column(scale=1):
                # Stats section
                gr.Markdown("### 📊 Performance Metrics")
                
                latency_display = gr.Markdown("**TTFT:** - ms\n\n**TPOT:** - ms")
                
                gr.Markdown("### 🎯 Expert Activation")
                
                expert_chart = gr.BarPlot(
                    x="Expert",
                    y="Count",
                    title="Token Distribution",
                    height=200,
                )
                
                refresh_stats = gr.Button("🔄 Refresh Stats")
        
        # Event handlers
        def generate_text(prompt, max_tokens, temperature, top_p):
            start = time.perf_counter()
            
            if hasattr(engine, 'generate_stream'):
                # Streaming generation
                output_text = ""
                for chunk in engine.generate_stream(prompt, max_tokens, temperature, top_p):
                    output_text += chunk
                    yield output_text
            else:
                output_text = engine.generate(prompt, max_tokens, temperature, top_p)
                yield output_text
            
            elapsed = time.perf_counter() - start
            # Update latency display (would need separate output)
        
        def update_stats():
            stats = engine.get_expert_stats()
            
            if "expert_distribution" in stats:
                import pandas as pd
                df = pd.DataFrame({
                    "Expert": [f"E{i}" for i in range(8)],
                    "Count": stats["expert_distribution"],
                })
                return df
            return None
        
        def clear_outputs():
            return "", None
        
        generate_btn.click(
            fn=generate_text,
            inputs=[prompt, max_tokens, temperature, top_p],
            outputs=[output],
        )
        
        clear_btn.click(
            fn=clear_outputs,
            outputs=[output, expert_chart],
        )
        
        refresh_stats.click(
            fn=update_stats,
            outputs=[expert_chart],
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Explain quantum computing in simple terms:"],
                ["Write a Python function to sort a list:"],
                ["The best way to learn machine learning is"],
                ["In the year 2050, humanity will"],
            ],
            inputs=[prompt],
        )
        
        gr.Markdown("""
        ---
        **Light-MoE** | [GitHub](https://github.com/Whitehall945/light-moe) | Apache 2.0 License
        """)
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Light-MoE Gradio Demo")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    demo = create_demo(args.model)
    demo.launch(server_port=args.port, share=args.share)
