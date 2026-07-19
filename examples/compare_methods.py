"""
Example comparing different speculative decoding methods.
"""

from seqspecdecod import load_models, benchmark

def main():
    # Load models
    print("Loading models...")
    tokenizer, draft_model, target_model = load_models(
        draft_name="Qwen/Qwen3-0.6B",
        target_name="Qwen/Qwen3-4B",
        device="cuda"
    )

    prompts = [
        "Give me a short introduction to large language models.",
        "Explain the difference between supervised and unsupervised learning.",
        "What is the transformer architecture and why is it so popular?",
        "What is gradient descent and how does it work?"
    ]

    methods = [
        ("standard", {}),
        ("naive_hsd", {}),
        ("capped_hsd", {"cap": 3}),
        ("capped_hsd", {"cap": 5}),
    ]

    results = {}

    for method_name, method_kwargs in methods:
        print(f"\n{'='*60}")
        print(f"Testing method: {method_name} {method_kwargs}")
        print(f"{'='*60}")

        baseline_results, spec_results = benchmark(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=200,
            gamma=5,
            method=method_name,
            verbose=True,
            **method_kwargs
        )

        results[f"{method_name}_{method_kwargs}"] = {
            "baseline": baseline_results,
            "speculative": spec_results
        }

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    for method_key, result_dict in results.items():
        spec_results = result_dict["speculative"]
        avg_speed = sum(r['tokens_per_sec'] for r in spec_results) / len(spec_results)
        avg_acceptance = sum(r['acceptance_rate'] for r in spec_results) / len(spec_results)
        avg_accepted_n = sum(r['average_accepted_n'] for r in spec_results) / len(spec_results)

        print(f"\n{method_key}:")
        print(f"  Average speed: {avg_speed:.1f} tokens/s")
        print(f"  Average acceptance rate: {avg_acceptance:.1%}")
        print(f"  Average accepted tokens: {avg_accepted_n:.2f}")

    # Compare against baseline
    baseline_results = results[list(results.keys())[0]]["baseline"]
    avg_baseline_speed = sum(r['tokens_per_sec'] for r in baseline_results) / len(baseline_results)

    print(f"\nBaseline average speed: {avg_baseline_speed:.1f} tokens/s")
    print("\nSpeedups vs baseline:")
    for method_key, result_dict in results.items():
        spec_results = result_dict["speculative"]
        avg_speed = sum(r['tokens_per_sec'] for r in spec_results) / len(spec_results)
        speedup = avg_speed / avg_baseline_speed
        print(f"  {method_key}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
