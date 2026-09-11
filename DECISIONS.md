# Design decisions

Short records of choices that were not obvious, with the alternative that lost.

## 1. Banking77 over a synthetic ticket dataset

Synthetic support datasets (for example the Bitext customer-support set) are clean and easy: a fine-tuned small model reaches 99 percent and the comparison teaches nothing. Banking77 is real user text with 77 intents that often differ by one detail (`card_arrival` versus `card_delivery_estimate`), so the gap between model sizes is visible.

## 2. Structured outputs with a label enum instead of free-text parsing

The frontier baseline constrains the response to a JSON schema whose `label` field is an enum of the 77 intents. The alternative, asking for the label as text and fuzzy-matching it, hides a class of errors (near-miss spellings, invented labels) inside the parser. With the enum, an invalid label is impossible and every mistake is a real routing mistake.

## 3. Low effort for the frontier baseline

Intent classification does not benefit from extended reasoning, and effort is a cost lever. The baseline runs at `effort: low`; the effort level is recorded in every result file so a higher-effort run can be compared later if the accuracy looks suspicious.

## 4. Disk cache keyed by prompt version

Every prediction is cached by `(model, effort, prompt version, example id)`. Re-running a report after a crash or a code change costs nothing, and bumping `PROMPT_VERSION` is the only way to invalidate results, which makes accidental prompt drift impossible to miss.

## 5. Fixed-seed sampling for partial runs

`router frontier --limit N` samples the same N rows every time, and every other system will be scored on the same rows. Comparing systems on different subsets is the most common way benchmark tables lie.

## 6. Cached predictions are excluded from latency

Latency percentiles are computed only from predictions made live in the current run. Mixing in cache hits would report zero-millisecond calls and make the frontier model look faster than it is.
