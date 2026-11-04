# UnFold Architecture

## Runtime

`app/app.py` presents a session-only Streamlit workspace. `src/assessment.py` validates the submitted profile, calls `predict_hybrid`, creates guidance exactly once, and returns a snapshot with an ID, UTC timestamp, input signature, result, guidance, action selections, and optional in-memory PDF.

The render loop reuses the snapshot. Input changes mark it stale, disable actions/report downloads, and require an explicit new submission. Draft journal and privacy choices survive navigation; checklist choices live in the snapshot. Clear drops the session data. Shared model caches contain model objects, never student assessments.

## Prediction and Support

1. `src/predict.py` maps canonical profile inputs to the saved estimator's selected features and exposes tabular SHAP explanations.
2. `src/nlp_engine.py` loads local RoBERTa weights and tokenizer. It returns raw softmax scores rather than arbitrarily compressed confidence. Prediction and token attribution use the same 256-token window. Attribution runs through the model's actual input embedding path using gradients only with respect to inputs; shared parameter gradients are not accumulated. Model access is serialized with a process lock.
3. `src/support_flags.py` applies explicit negation and contextual rules. Language flags remain heuristic conversation prompts. Historical/third-person references are not automatically labeled current crises.
4. `src/fusion.py` combines the uncalibrated scores with fixed 0.45/0.55 weights. Missing text is an absent modality, not an invented neutral score. Support flags never change model scores. Agreement/disagreement labels describe the direction of the numeric difference without inferring emotional masking or clinical certainty.

## Guidance and Reporting

`src/genai_advisor.py` provides local guidance by default. Explicit online opt-in is required even if a key is present. A structured Gemini request supplies student actions and counselor sections together. Server errors are retried once by the SDK; server failure, timeout, or model-not-found may switch to one configured fallback model. Each of at most two models is limited to two attempts with a 15-second per-request timeout. Authentication, quota, and request-configuration errors stop immediately. Responses are validated; missing configuration or exhausted attempts return labeled local guidance. Potential current support-language flags use local support guidance without forwarding the journal.

`src/report_generator.py` returns PDF bytes by default. Export happens only when requested, and the result is reused. Reports include the submitted profile, full reflection, separate support notes, and intake brief. The caller can explicitly pass an output path for a deliberate file export. Reports paginate instead of silently truncating reflections to one page.

Clear removes session references, not downloaded files or data already transmitted to an opted-in provider. No claim of guaranteed physical memory erasure is made.

## Evidence and Evaluation

`src/research.py` reads actual tabular evaluation metadata from the model bundle. Historical journal training plots are not treated as paired validation. Hybrid metrics are not published until a paired evaluation artifact exists.

`src/evaluate_hybrid.py` accepts canonical profile columns plus a reflection and binary label for each held-out student. It compares profile, journal, and fused scores on the same rows, writes only aggregate metrics and provenance, and rejects empty/malformed paired data. The researcher must ensure both models' training data are excluded. Probability calibration requires separate fitting and validation; neither fixed fusion nor Brier score reporting establishes calibrated personal risk.

Tabular baselines currently use a smaller training subset; comparisons must retain that qualification. `docs/report_content.md` and `docs/ppt_content.md` describe the earlier tabular project and are legacy drafts, not evidence for hybrid performance.
