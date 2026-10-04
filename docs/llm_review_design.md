# Proposer/reviewer offline LLM design

`run_llm_review_design.py` is an additional offline comparison workflow. A
proposer model writes a complete shared-feature candidate, the host validates
that candidate on every sample in the selected fixed-sample artifact, and a
separate reviewer model reviews the validated implementation. It does not run
the Lipschitz evaluator, training, or environment evaluation.

The host approves only when the exact candidate version reviewed by the
reviewer has a passing program-validation report and the reviewer returns a
valid `needs_revision=false` response. The approved artifact records
`approval_method=model_review` and
`lipschitz_evaluation_performed=false`. It uses the existing shared-feature-v2
runtime and can later be passed explicitly to `td3_dinkelbach_llm`; model review
does not establish that training performance will improve.

## Limits and resume

`--max-code-repairs 5` means one initial proposer output plus at most five
repairs in each proposal cycle. Parse, static, or execution failures consume
that repair allowance but do not consume a review round. A valid reviewer
response completes one review round. When the reviewer requests changes, the
next proposal cycle receives a fresh repair allowance.

`--max-review-rounds 5` permits five completed reviews. If the fifth review
still requests changes, its suggestions are saved and the run pauses without
generating an unsendable sixth candidate. Resume preserves all counters and
phases. Supplying an explicitly larger limit permits work to continue; limits
cannot be reduced. Invalid, truncated, missing, mismatched-model, or transport-
failed review responses never approve a candidate.

The default output root is `results/llm_review_designs/`. Each call and each
candidate has its own saved prompt, planned request, response, validation, and
review record. No API key is written into these files.

## PowerShell examples

No additional package is required beyond the existing LLM_HRL environment.

Dry run (builds and saves both full prompts but sends no request):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_review_design.py `
  --fixed-sample 'results/llm_baselines/baseline-20260927T130708Z-ba4bf287' `
  --proposer-provider lmstudio `
  --proposer-base-url 'http://127.0.0.1:1234/v1' `
  --proposer-model 'qwen/qwen3.8-27b' `
  --proposer-context-length 50000 `
  --proposer-max-output-tokens 4096 `
  --reviewer-provider openai `
  --reviewer-model 'gpt-4o' `
  --reviewer-context-length 50000 `
  --reviewer-max-output-tokens 4096 `
  --dry-run
```

Formal offline design:

```powershell
$env:OPENAI_API_KEY = '<your-key>'
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_review_design.py `
  --fixed-sample 'results/llm_baselines/baseline-20260927T130708Z-ba4bf287' `
  --proposer-provider lmstudio `
  --proposer-base-url 'http://127.0.0.1:1234/v1' `
  --proposer-model 'qwen/qwen3.8-27b' `
  --proposer-context-length 50000 `
  --proposer-max-output-tokens 4096 `
  --proposer-temperature 0.3 `
  --proposer-seed 20260927 `
  --reviewer-provider openai `
  --reviewer-model 'gpt-4o' `
  --reviewer-context-length 50000 `
  --reviewer-max-output-tokens 4096 `
  --reviewer-temperature 0.3 `
  --reviewer-seed 20260927 `
  --max-review-rounds 5 `
  --max-code-repairs 5 `
  --beta 1.0
```

Resume without changing limits:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_review_design.py `
  --resume 'results/llm_review_designs/<models>/<run-id>'
```

Resume while explicitly increasing either allowance:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_review_design.py `
  --resume 'results/llm_review_designs/<models>/<run-id>' `
  --max-review-rounds 7 `
  --max-code-repairs 8
```

The resume contract requires the same Git revision and fixed-sample content.
It restores the saved provider, model, endpoint, beta, candidate, reviewer
suggestions, phase, and counters rather than accepting silent replacements.
