# Full-attention evaluation capacity

The paired experiment evaluates up to 16 concurrent native 4×4 games per B200. Its fixed controller samples at most 128 thinking tokens and 32 final-answer tokens per response, inserts the 11-token thinking transition, and ends after at most 96 responses. The model context limit and full causal attention remain 262,144 tokens.

A CPU max-plus audit of both flip phases and all three retry states bounds the complete transcript by 53,568 tokens. It charges every native feedback UTF-8 byte as one token, uses a 512-token upper bound for the immutable system prefix, overcounts user and assistant role delimiters, and includes feedback after the last response. Native board renderings have exactly 74 ASCII bytes for every visibility pattern. The fixed native revision is `e3d195e857727dfa3d4a96561be202098d67da83`.

| Native feedback | Conservative UTF-8 bytes |
|---|---:|
| First-pick observation | 280 |
| Second-pick observation | 206 |
| Both-card feedback | 162 |
| Retry | 221 |
| First-pick forfeit | 129 |
| Second-pick forfeit | 156 |

Each feedback message receives eight extra template tokens. Each assistant response is charged 179 tokens: 128 thinking, 32 answer, 11 forced transition, and eight template tokens. A six-state dynamic program explores all valid-flip and invalid-attempt transitions for 96 responses, including the native forfeit feedback and next observation after the third failed attempt.

The model uses 147,456 KV bytes per token. Rounding each bounded transcript to a 128-token page and adding one spare page yields 53,760 reserved tokens per game. Sixteen games reserve 126,835,752,960 bytes per GPU. The existing four-game full evaluator measured a 154,694,320,128-byte pool on each 191,514,411,008-byte B200. The new reservation is 81.99% of that previously allocated pool; 32 bounded games would exceed it.

The audit ran without a CUDA context or model pass. Its source SHA256 is `18ff46a4854bd7e3c3662587196187ef6af4d1693bf9353c3cb972bb2d370979`; its saved result is `rng-hero-paired-a1/diagnostics/full-evaluation-transcript-kv-bound-a1.json`. The source and result are preserved with experiment artifacts. This is a capacity bound, not a measured speedup or a numerical likelihood-parity result.

The evaluation CLI accepts `--evaluation-concurrency 16 --full-kv-token-bound 53568` only for full-attention evaluation with this controller. The paired final executor supplies these operational arguments without modifying canonical condition configurations, checkpoint selection, board seeds, rewards, or generation limits. Complete episode records are reusable across physical concurrency changes. Interrupted cohort receipts remain part of compute accounting.
