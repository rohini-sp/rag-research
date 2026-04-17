# run notes — claude cli backend branch

branch: `feat/claude-cli-backend`

## model set for full experiment
- gpt-5 (openai)
- gpt-5-mini (openai)
- o4-mini (openai)
- claude sonnet 4.6 cli (`provider=claude_cli`, `model=sonnet`)
- claude opus latest cli (`provider=claude_cli`, `model=opus`)
- gemini 2.5 flash (gemini)
- llama 3.3 70b (groq)
- llama 3.1 8b (groq)

## retry policy
- no retry for model ids starting with `gpt-`
- retry (up to 3 attempts) for all other models
- applied to both extraction and qa stages

## anthropic cost guard
- anthropic api provider is not used for claude runs
- claude models run via local `claude_cli` backend only
