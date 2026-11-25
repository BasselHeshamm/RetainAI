# RetainAI

A chat assistant for a telecom retention team. You ask it a question in plain English, like "which customers are most likely to leave and what should we offer them?", and it answers using real customer data, a churn prediction model and the company's offer policies.

I built it to practice the things I care about most in applied AI: classical ML that is properly validated, LLM agents that call tools, and retrieval over documents (RAG), all wired together into something a non-technical person could actually use.

## What it does

Behind the chat there is a small team of agents:

- **Supervisor.** Reads the question, decides which specialist should work next, and stops when it has enough to answer.
- **Analytics agent.** Writes its own SQL against the customer database to answer questions like "what is the churn rate by contract type?"
- **Churn scoring agent.** Uses an XGBoost model to say how likely a customer is to leave, ranks customers by revenue at risk, and explains each prediction with SHAP.
- **Offers agent.** Searches the retention policy documents (offer catalogue, approval rules, contact guidelines) and recommends offers the customer is actually eligible for.
- **Responder.** Writes the final answer using only what the agents found.

```
question --> supervisor --+--> analytics agent ------> SQLite (customers + scores)
                ^         +--> churn scoring agent --> XGBoost + SHAP
                |         +--> offers agent ---------> Chroma (policy documents)
                |                    |
                +---- findings ------+
                |
                +--> when finished --> responder --> answer
```

A typical multi-step question: "Find the active customer with the most revenue at risk and recommend an offer." The supervisor sends the churn scoring agent first, it finds the customer and why they are at risk, the supervisor then passes that customer id to the offers agent, which checks eligibility against the policies, and the responder puts it together.

## Results

| | |
| --- | --- |
| Data | IBM Telco Customer Churn dataset, 7,043 customers, 26.5% churn |
| Churn model | XGBoost, hold-out ROC-AUC **0.843** (PR-AUC 0.658) |
| Agent benchmark | 20 questions with known answers: **20/20** routed to the right agents, **20/20** answered correctly on the latest evaluation |

The benchmark checks two things for every question: did the supervisor call the right agents, and does the answer contain the correct facts. Expected numbers are computed from the database at test time, so the test can't drift away from the data.

## Things I learned building it

**The evaluation caught real bugs.** The first run scored 60%. Most failures turned out to be my scoring being too strict (an answer of 11.27% marked wrong because I expected 11.3, and the model writing customer ids with a special hyphen character). Once those were fixed, the interesting failure remained: on two questions the supervisor finished early and the final answer invented a retention offer from the model's general knowledge. I fixed that with a guardrail in code: any question about offers, eligibility, approvals or contact rules cannot finish until the offers agent has checked the policy documents. That one isn't left to the LLM's judgment.

**Stored scores are out-of-fold.** Every customer's churn probability comes from 5-fold cross-validation, so no score was produced by a model that saw that customer during training. Scoring the training data directly would look better than it really is.

**SHAP is more useful when it speaks the business's language.** One-hot encoding splits "contract" into several columns. I sum the SHAP values back per original column, so the agent says "contract: Month-to-month raises risk" instead of listing encoded feature names.

**The SQL agent can only read.** It can run a single SELECT on a read-only connection. Anything else is rejected before it reaches the database.

**Free tiers shape the design.** Rate limits on the free LLM tier made me shorten tool outputs, add retries, and let the benchmark re-run only failed questions.

## Running it

You need Python 3.11+ and a free API key from [Groq](https://console.groq.com/keys) (Gemini and OpenAI also work).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then paste your key after GROQ_API_KEY=
python scripts/setup_data.py      # downloads the data, trains the model, builds the search index
streamlit run app.py              # the chat app
```

Other useful commands:

```bash
pytest -q                                # tests, no API key needed
python scripts/evaluate.py               # the 20-question benchmark
python scripts/evaluate.py --failed      # re-run only the questions that failed last time
uvicorn api:app --reload                 # REST API: POST /ask, GET /customers/{id}/risk
```

On a Mac, if XGBoost complains about `libomp.dylib`, run `brew install libomp`.

## Project layout

```
src/retainai/
  db.py        loads the CSV into SQLite, read-only query guard
  model.py     features, XGBoost training, out-of-fold scores, SHAP explanations
  rag.py       splits the policy documents into sections and searches them with Chroma
  tools.py     the six tools the agents can call
  agents.py    supervisor, specialist agents, responder, and the LangGraph wiring
  llm.py       picks Groq, Gemini or OpenAI depending on which key is set
docs/policies/ the retention policy documents used for search
eval/          benchmark questions
tests/         tests for the data, model, SQL guard, search, routing and API
app.py         Streamlit chat app
api.py         FastAPI service
```

## Limitations

- The policy documents are written by me for this project. The company and its offers are fictional.
- The dataset is a single snapshot, so there is no time-based validation of the churn model.
- A churn model predicts who will leave, not who an offer will actually keep. Uplift modelling would be the natural next step.
