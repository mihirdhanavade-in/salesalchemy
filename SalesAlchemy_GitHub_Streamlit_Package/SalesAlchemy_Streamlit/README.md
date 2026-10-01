# SalesAlchemy — Streamlit Community Cloud edition

Turn customer requirements into winning proposals.

This is the GitHub/Streamlit deployment edition of the TEC 406 SalesAlchemy capstone. It retains the five-section proposal, Python pricing, retrieved sources, prompt-quality scorecard, A/B comparison, human ratings and DOCX/Markdown downloads.

## Start here

1. Extract the deployment ZIP on your computer.
2. Create a GitHub repository named `salesalchemy`. Start with **Private** visibility for a classroom trial.
3. Upload the **contents** of the extracted `SalesAlchemy_Streamlit` folder. `app.py`, `engine.py` and `requirements.txt` must be at the repository root. Keep `kb/` and `.streamlit/` as folders. Do not upload only the ZIP or the Colab notebook.
4. Commit to branch `main`.
5. Sign in at https://share.streamlit.io and connect the GitHub account that owns the repository. Grant access to this repository if it is private.
6. Choose **Create app → Yup, I have an app**. Select your repository, branch `main`, and entrypoint `app.py`.
7. Under **Advanced settings**, select **Python 3.12**. Leave Secrets empty for the initial demo deployment.
8. Click **Deploy**. Open the resulting `streamlit.app` address and generate a proposal in Demo mode.

Do not run the notebook's Colab launch cells, Cloudflare tunnel, localtunnel or `nohup` commands on Streamlit Community Cloud. Community Cloud starts the app itself.

## Deployment settings

| Setting | Value |
|---|---|
| Suggested repository | `salesalchemy` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | `3.12` |
| Dependency file | `requirements.txt` |
| Initial mode | Demo; no token required |
| Live generation | Hugging Face Inference Providers; separate configuration required |

## Why this version differs from Colab

The Colab notebook loads a multi-billion-parameter language model into its own RAM/GPU. That setup is unsuitable for ordinary Community Cloud resources. The 1.5B CPU model alone needs roughly 6 GB for float32 parameters, before runtime overhead. This deployment edition loads no language-model weights and does not install PyTorch, Transformers or sentence-transformers.

| Component | Original Colab edition | This deployment edition |
|---|---|---|
| LLM | Local Qwen/SmolLM model | Optional hosted chat model selected by the owner |
| Retrieval | MiniLM with TF-IDF fallback | Local TF-IDF cosine similarity |
| Sampling | Local Transformers parameters | Hosted temperature, top-p and token limit |
| Pricing | Decimal arithmetic | Same Decimal arithmetic |
| Access | Temporary tunnel URL | Streamlit-managed URL and viewer settings |
| Demo | Canned output | Same canned output; no inference requests |

Local-only options such as `torch.inference_mode`, local chat templating and `repetition_penalty` do not apply to this hosted adapter. A high-level chat request is sent to the provider instead. Update your project submission's deployment/RAG description if you use this variant; do not claim this cloud version runs a local GPU model or MiniLM embeddings.

## Enable live AI after the demo works

1. Create a Hugging Face token with **Inference Providers** permission at https://huggingface.co/settings/tokens. Do not send the token in chat or commit it to GitHub.
2. Open https://huggingface.co/playground and select a currently supported **text chat, preferably non-reasoning** model. Check its licence, provider availability, account access and pricing. A model repository being public does not mean a hosted provider serves it. `Qwen/Qwen2.5-7B-Instruct-1M` is an example to check, not a guaranteed endpoint.
3. In your Streamlit app's **Settings → Secrets**, add:

```toml
HF_TOKEN = "your_actual_hugging_face_token"
HF_MODEL_ID = "your_selected_supported_model_id"
```

4. Save and restart the app if needed. **Demo mode** becomes switchable once both settings are present. Turn it off to generate with the provider.

No inference requests occur on startup. Generating a proposal makes six model requests; A/B comparison makes seven. Hosted generation may consume credits or incur charges. Inspect your Hugging Face usage and billing settings; this app does not impose an account-wide spending cap. Keep paid live access restricted to trusted viewers using Streamlit's access controls.

Live generation sends the customer brief, retrieved document excerpts and calculated pricing to Hugging Face and its selected provider. Those services' processing and retention terms apply. Demo mode does not send these inputs to a model provider.

The adapter uses the fixed HTTPS Hugging Face router endpoint, a request timeout, safe error messages and no automatic retries. A timed-out request may still have been processed by the provider; check usage before repeatedly retrying.

## Files included

| Path | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `engine.py` | Retrieval, pricing, prompts, remote generation, evaluation and export |
| `requirements.txt` | Lightweight Python dependencies |
| `.streamlit/config.toml` | Brand theme and upload configuration |
| `.streamlit/secrets.toml.example` | Blank configuration example; no real credentials |
| `.gitignore` | Excludes secrets, caches and local runtime files |
| `kb/company_profile.md` | Fictional sample company facts |
| `kb/services_and_rates.json` | Authoritative illustrative INR prices |
| `kb/case_studies.md` | Fictional project patterns |
| `kb/terms_and_faq.md` | Payment terms, validity and exclusions |
| `tests/test_engine.py` | Offline pricing, retrieval, export and adapter checks |
| `tests/test_ui.py` | Streamlit interface smoke test in Demo mode |

## Uploading knowledge documents

For a `.txt` or `.md` upload, leave **Replace the fictional sample KB** unchecked. The uploaded text supplements the included sample; prices continue to come from the JSON rate card.

For a full replacement, upload your company documents **and** a file named exactly `services_and_rates.json`, then enable replacement. Follow the existing JSON schema. Set the currency selector to the rate card's currency. Text-file prices do not change calculated prices. There is no automatic currency conversion.

## Local run and tests

From the repository folder:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
streamlit run app.py
```

Use Python 3.12. To enable live mode locally, create `.streamlit/secrets.toml` using the example. The real secrets file is excluded from Git.

If using Git from a terminal after creating an empty repository:

```bash
git init -b main
git add .
git commit -m "Prepare SalesAlchemy for Streamlit deployment"
git remote add origin https://github.com/YOUR_USERNAME/salesalchemy.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual account. Authenticate using GitHub's supported sign-in flow; never embed a token in the remote URL.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `app.py` cannot be found | Upload the extracted folder contents to the repo root, or correct the entrypoint path. |
| Missing package | Confirm `requirements.txt` is committed alongside `app.py`, then reboot. |
| Only Demo mode is available | Both `HF_TOKEN` and `HF_MODEL_ID` must be set in Secrets. |
| HTTP 401/403 | Token validity, provider permission, and model access. |
| HTTP 400/404 | Model support in the current Inference Playground and exact model ID. |
| HTTP 402/429 | Account credits/billing or rate limits; switch to Demo as needed. |
| Response reaches token limit | Increase the cap, shorten the brief, or choose a non-reasoning model. |
| Replacement KB needs JSON | Untick replacement for text-only uploads, or add the required rate card. |
| Download disabled | Review the draft and tick the review checkbox. |
| Changed code is not visible | Confirm the deployed branch, inspect logs, and reboot. |

## Governance and test status

All included company data and rates are fictional. Outputs require factual and commercial review. Automated scores are heuristic screening results, not proof of accuracy. No email is sent. The application does not create a customer database; session memory and user downloads still contain submitted data. Hosting and inference infrastructure can process data under their own policies.

Offline validation is recorded in `VALIDATION.md`. Actual GitHub publication, Streamlit deployment and a live provider call require the owner's account configuration and are not claimed as completed by this package.

## Primary references

- Deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- Resource limits: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- Hosted chat API: https://huggingface.co/docs/inference-providers/tasks/chat-completion
- Provider pricing: https://huggingface.co/docs/inference-providers/pricing
