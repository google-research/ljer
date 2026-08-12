"""Central configuration. Everything needed to reproduce the run lives here.

Checkpoints are pinned to commit SHAs, verified with provenance/hf_probe.py.

A note on Hugging Face. This project READS from Hugging Face and writes nothing
there. Model weights and the VerifyBench benchmark are downloaded from their
public repositories at run time, in the same sense that a package manager
downloads a dependency. No dataset, model, or result is uploaded to Hugging Face.
Released artifacts live in this repository only.
"""
import os

# ---------------------------------------------------------------- models
# Served with vLLM offline batch on a rented GPU (NOT a hosted inference API).
# Rationale: serverless providers deprecate and silently REDIRECT small models
# (DeepInfra redirects Llama-3.2-1B -> 3B), which would corrupt results without
# any error. HF checkpoints are permanent and exactly specifiable.
#
# `pred_sum` = predicted eps_0 + eps_1 from Yan et al. Table 2 accuracy on
#   VerifyBench, where pi = 0.5 exactly so eps_0 + eps_1 = 2*(1 - accuracy).
#   PRE-REGISTERED. These values were fixed BEFORE any data was collected and
#   have not been revised. They are the pre-registration record that makes
#   predicted-versus-observed a genuine out-of-sample comparison, so editing them
#   would invalidate that comparison. Treat as read-only. None = no anchor.
# `dtype`: bfloat16 everywhere except gpt-oss-20b, which ships natively
#   MXFP4-quantized; we serve every model in its RELEASED precision.
MODELS = [
    dict(key="gemma3-1b",   repo="google/gemma-3-1b-it",
         rev="dcc83ea841ab6100d6b47a070329e1ba4cf78752",
         dtype="bfloat16", gb=2.0,  license="gemma",      pred_sum=0.947),
    dict(key="qwen3-1.7b",  repo="Qwen/Qwen3-1.7B",
         rev="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
         dtype="bfloat16", gb=4.1,  license="apache-2.0", pred_sum=0.378),
    dict(key="qwen3-8b",    repo="Qwen/Qwen3-8B",
         rev="b968826d9c46dd6066d109eabc6255188de91218",
         dtype="bfloat16", gb=16.4, license="apache-2.0", pred_sum=0.120),
    dict(key="qwen35-9b",   repo="Qwen/Qwen3.5-9B",
         rev="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
         dtype="bfloat16", gb=19.3, license="apache-2.0", pred_sum=None),
    dict(key="gptoss-20b",  repo="openai/gpt-oss-20b",
         rev="6cee5e81ee83917806bbde320786a8fb61efebee",
         dtype="auto",     gb=41.3, license="apache-2.0", pred_sum=0.104),
    dict(key="gemma4-26b",  repo="google/gemma-4-26B-A4B-it",
         rev="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
         dtype="bfloat16", gb=51.6, license="apache-2.0", pred_sum=None),
    dict(key="qwen3-32b",   repo="Qwen/Qwen3-32B",
         rev="9216db5781bf21249d130ec9da846c4624c16137",
         dtype="bfloat16", gb=65.5, license="apache-2.0", pred_sum=0.084),
]
# gemma-4 license VERIFIED 2026-08-07: license_link on the model card resolves to
# verbatim Apache License 2.0 text. Google changed licensing between Gemma 3 and
# Gemma 4 (Gemma 3 remains under the Gemma Terms of Use and is gated).
# Roster licensing: 6x Apache-2.0, 1x Gemma (gemma-3-1b-it).

# Reasoning control, verified per model by rendering the chat template.
#   Qwen3 / Qwen3.5 : enable_thinking=False injects an empty <think></think>
#                     block. Confirmed working on all four.
#   gpt-oss-20b     : harmony format, ALWAYS reasons; only the effort level is
#                     tunable. Set to "low".
#   gemma-4-26B     : emits a <|channel>thought block; no documented off switch.
#                     We accept it and disclose that two judges reason natively.
# Rationale for minimizing reasoning: it matches the cheap high-volume judge
# case, keeps outputs short, and -- decisively -- extended reasoning suppresses
# the sampling variance this method consumes.
CHAT_KWARGS = {
    "qwen3-1.7b": {"enable_thinking": False},
    "qwen3-8b":   {"enable_thinking": False},
    "qwen35-9b":  {"enable_thinking": False},
    "qwen3-32b":  {"enable_thinking": False},
    "gptoss-20b": {"reasoning_effort": "low"},
    "gemma3-1b":  {},
    "gemma4-26b": {},
}
# Models that reason regardless of settings -- report separately, and expect
# longer outputs and possibly lower within-item variance.
NATIVE_REASONERS = {"gptoss-20b", "gemma4-26b"}

# ---------------------------------------------------------------- sampling
# NOTE: Qwen generation_config defaults are temp=0.6/top_p=0.95. We override to
# 0.7/0.9 for consistency across the roster and with our prior work. Disclose.
N_ROUNDS    = 10
TEMPERATURE = 0.7
TOP_P       = 0.9
MAX_MODEL_LEN = 24576  # vLLM context ceiling = longest prompt + MAX_TOKENS.
                       # VerifyBench completions carry <think> traces; measured
                       # prompts reach ~17k tokens. vLLM pages KV cache, so a
                       # generous ceiling costs concurrency only when long
                       # prompts actually appear.
MAX_TOKENS  = 2048    # was 700 -- too low: gpt-oss and gemma-4 reason before
                      # answering and would be cut off before "Final Judgment:".
                      # vLLM stops at EOS, so non-reasoning models still emit
                      # ~300 tokens; the higher cap costs nothing for them.
PILOT_VB    = 150     # VerifyBench      (pi = 0.500, balanced -> rate estimates)
PILOT_VBH   = 100     # VerifyBench-Hard (pi = 0.291, dispersion check)
SEED        = 20260807

# ---------------------------------------------------------------- prompts
# ARM A -- verbatim, Yan et al. Appendix G.2. Do not alter a character.
PROMPT_VERBATIM = """Given the following math problem and the reference answer. Judge the correctness of the answers given later, with some ability to generalize and match the form and format of the answer results. The following specific requirements are followed when judging:

1. Judge only whether the final result of the reference answer and the answer to be judged agree; do not consider whether there are any errors in the process. Don't verify the correctness of the answer by yourself, please only refer to the reference answer for the correctness of the answer.
2. The reference answer and the answer to be judged only need to be essentially the same, ignoring irrelevant details such as units, symbols, whether or not to approximate, and the form of expression in the answer. The two answers are considered to be consistent if they are equivalently transformable.
3. All your analysis answer must be in English.
4. Please analyze the judged answer and try to compare it with the reference answer.
At the end of all analysis, give the result of the judgment on an extra line at the end of the answer in the form "Final Judgment: Yes/No".

Problem: {question}
Reference Answer: {answer}
Solution to be evaluated: {completion}"""

# ARM B -- verbatim plus one line eliciting the difficulty covariate H_c.
PROMPT_DIFFICULTY = PROMPT_VERBATIM + """

After the judgment line, give the difficulty of this verification task on an extra line in the form "Difficulty: N", where N is an integer from 1 (very easy; the answers obviously match or obviously differ) to 5 (very difficult; ambiguous, requires subtle equivalence reasoning, or the correct call is genuinely unclear)."""

# ARM C -- verbatim, Yan et al. Appendix G.3 (no reference answer supplied).
# A published ablation costing 5-18% accuracy; our principled lever into the
# eps_0 + eps_1 >= 1 regime, and a real deployment case (no gold answer to hand).
PROMPT_NOREF = """Given the following math problem, please judge the correctness of the answers given later. The following specific requirements are followed when judging:

1. All your analysis answer must be in English.
2. Please analyze the math problem and the answer and try to tell whether the given completion is a correct answer. At the end of all analysis, give the result of the judgment on an extra line at the end of the answer in the form " Final Judgment: Yes/No ".

Problem: {question}
Solution to be evaluated: {completion}"""

ARMS = {"verbatim": PROMPT_VERBATIM, "difficulty": PROMPT_DIFFICULTY, "noref": PROMPT_NOREF}

# ---------------------------------------------------------------- field map
FIELDS = dict(question="question", answer="answer", completion="completion", gold="gold_correct")
