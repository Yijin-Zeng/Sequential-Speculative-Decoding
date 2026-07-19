"""
Basic usage example for the Sequential-Speculative-Decoding package.
"""
from seqspecdecod import load_models, speculative_generate, benchmark

def main():
    # Load models
    print("Loading models...")
    tokenizer, draft_model, target_model = load_models(
        draft_name="Qwen/Qwen3-0.6B",
        target_name="Qwen/Qwen3-4B",
        device="cuda"
    )

    # Single generation example
    print("\n=== Single Generation Example ===")
    result = speculative_generate(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        prompt="Explain what is machine learning in one paragraph.",
        max_new_tokens=200,
        gamma=5,
        method="standard"
    )

    print(f"\nGenerated text:\n{result['text']}")
    print(f"\nStats:")
    print(f"  Tokens generated: {result['tokens_generated']}")
    print(f"  Speed: {result['tokens_per_sec']:.1f} tokens/s")
    print(f"  Acceptance rate: {result['acceptance_rate']:.1%}")
    print(f"  Average accepted tokens: {result['average_accepted_n']:.2f}")

    # Benchmark example
    print("\n=== Benchmark Example ===")
    prompts = [
        "Give me a short introduction to large language models.",
        "Explain the difference between supervised and unsupervised learning.",
    ]

    baseline_results, spec_results = benchmark(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=200,
        gamma=5,
        method="standard",
        verbose=True
    )

    # Summary
    print("\n=== Summary ===")
    avg_baseline_speed = sum(r['tokens_per_sec'] for r in baseline_results) / len(baseline_results)
    avg_spec_speed = sum(r['tokens_per_sec'] for r in spec_results) / len(spec_results)
    avg_speedup = avg_spec_speed / avg_baseline_speed
    avg_acceptance = sum(r['acceptance_rate'] for r in spec_results) / len(spec_results)

    print(f"Average baseline speed: {avg_baseline_speed:.1f} tokens/s")
    print(f"Average speculative speed: {avg_spec_speed:.1f} tokens/s")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Average acceptance rate: {avg_acceptance:.1%}")

if __name__ == "__main__":
    main()
