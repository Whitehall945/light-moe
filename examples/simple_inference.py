"""
Simple inference example using Light-MoE.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Light-MoE Inference Example")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the theory of relativity in simple terms:",
        help="Input prompt",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    # Import here to allow --help without CUDA
    from light_moe import LightMoEEngine

    print(f"Loading model from {args.model_path}...")
    engine = LightMoEEngine(
        model_path=args.model_path,
        tensor_parallel_size=1,
        expert_parallel_size=1,
    )

    print(f"\nPrompt: {args.prompt}\n")
    print("Generating...")
    
    output = engine.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    print(f"\nOutput:\n{output}")
    
    engine.shutdown()


if __name__ == "__main__":
    main()
