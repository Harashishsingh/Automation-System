# AI-Powered Customer Support Automation

A Django-based customer-support ticket management application that uses machine-learning models to classify incoming support messages, estimate their priority, recommend an action, and apply configurable automation rules before updating ticket status.

The project combines a web dashboard, ticket-management workflows, a TF-IDF + classical machine-learning classification pipeline, and a rule-based decision layer. It also includes scripts for preparing and training the models.

> **Scope note:** This README documents the code included in the supplied `Automation.zip`. Some settings and interface models (such as watched inboxes and notification preferences) are represented in the application, but the current source does not implement a live email integration or notification delivery service.

## Contents

- [Key features](#key-features)
- [How the application works](#how-the-application-works)
- [Machine-learning pipeline](#machine-learning-pipeline)
- [Decision and automation logic](#decision-and-automation-logic)
- [AI response generation](#ai-response-generation)
- [Application pages and routes](#application-pages-and-routes)
- [Data models](#data-models)
- [Project structure](#project-structure)
- [Technology stack](#technology-stack)
- [Requirements](#requirements)
- [Setup and run locally](#setup-and-run-locally)
- [Train or rebuild the models](#train-or-rebuild-the-models)
- [Database and Django administration](#database-and-django-administration)
- [Configuration](#configuration)
- [Current limitations and production considerations](#current-limitations-and-production-considerations)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

## Key features

### Ticket management
- Create support tickets with a title, customer name, and message.
- Run each new ticket through the classification and decision pipeline.
- Browse tickets and search by title, message, or customer name.
- Filter tickets by predicted category and status.
- Sort tickets by newest, oldest, or priority.
- Open an individual ticket and manually change its status.
- Record whether a human has overridden the automated status.

### AI-assisted classification
Three separately trained models predict:
- **Category** — the broad support area.
- **Intent** — the customer's more specific request.
- **Priority** — high, medium, or low.

The classifier combines word-level and character-level TF-IDF features. It calculates a margin-based confidence estimate for each prediction and combines those scores into an overall confidence value.

### Rule-based automation guardrails
- Create automation rules with a name, optional category, optional priority, and action.
- Activate or pause rules.
- Delete rules.
- Use blank category or priority values as wildcards.
- Override the machine-learning recommendation when an active rule matches.
- Track how often each rule has been triggered.

Supported actions:
- `resolve` — mark the ticket as resolved.
- `draft` — leave the ticket open for review.
- `escalate` — mark the ticket as escalated.

### Decision history
Each processed ticket creates a `DecisionLog` record containing the classification, priority, ML-recommended action, confidence, matched rule (if any), final action, reason, and timestamp.

### Dashboard and settings
- Dashboard counts for total, open, resolved, and escalated tickets.
- Recent-ticket overview.
- Automation-rule management page.
- Decision-log page with action counts and human-override count.
- Organization identity, support email, timezone, and business-hour fields.
- Agent active/inactive setting.
- Watched-inbox records and notification preference fields.
- Django admin registrations for the core data models.

### Test ticket generation
The application can choose a random customer message from `ml_training/cleaned_dataset.csv`, create a test ticket, and send it through the same processing pipeline as a normal ticket.

## How the application works

The main ticket-processing flow is:

```text
User submits a ticket
        |
        v
Django creates a Ticket record
        |
        v
classifier.py
  - word TF-IDF transform
  - character TF-IDF transform
  - category prediction
  - intent prediction
  - priority prediction
  - confidence calculation
        |
        v
response_generator.py
  - creates a category/intent-based draft response
        |
        v
engine.py
  - checks active AutomationRule records
  - applies rule / confidence decision policy
        |
        v
Ticket is updated
  - category, intent, priority
  - ML action and confidence
  - final action and reason
  - generated response
  - status
        |
        v
DecisionLog record is created
        |
        v
Ticket list / detail / dashboard
```

The ML model makes a prediction; it does not independently enforce the final action. The decision engine is a separate layer that checks configured rules and applies its own confidence thresholds.

## Machine-learning pipeline

The model-training code is in `ml_training/`.

### Dataset preparation

`clean_dataset.py` reads `dataset.csv` and:
1. Keeps the `instruction`, `category`, and `intent` columns.
2. Removes rows with missing required values.
3. Normalizes message text to lowercase, removes punctuation, and collapses whitespace.
4. Removes messages shorter than six characters.
5. Maps source categories to normalized category names.
6. Derives priority from the intent mapping.
7. Derives a reference action from priority.
8. Removes duplicate messages.
9. Writes `cleaned_dataset.csv`.

The cleaned dataset contains these columns:

| Column | Meaning |
|---|---|
| `message` | Normalized customer message used as model input |
| `category` | Mapped support category |
| `intent` | Specific customer intent |
| `priority` | Derived priority label |
| `action` | Reference action derived during data preparation |

The training script uses `message`, `category`, `intent`, and `priority` as its features/targets. The `action` column is not a separately trained target in the current training pipeline.

### Category mapping and priority labels

`intent_mapping.py` maps source categories to normalized values such as `account`, `order`, `refund`, `invoice`, `payment`, `delivery`, and `subscription`.

Priority is assigned from intent groups:
- **High:** refund requests, order cancellation, payment issues, complaints, delivery problems, cancellation-fee checks, and requests for a human agent.
- **Medium:** order tracking/changes, shipping-address changes, account edits, refund tracking, password recovery, and registration problems.
- **Low:** intents not included in the high- or medium-priority sets.

The exact mapping is defined in `ml_training/intent_mapping.py`.

### Feature extraction

`train_model.py` creates two TF-IDF feature representations:

**Word TF-IDF**
- Lowercases text.
- Uses unigrams and bigrams (`ngram_range=(1, 2)`).
- Uses English stop-word removal.
- Limits the vocabulary to 20,000 features.
- Uses sublinear term frequency.

**Character TF-IDF**
- Uses character n-grams of length 3 to 5.
- Limits the vocabulary to 20,000 features.
- Uses sublinear term frequency.

The two sparse matrices are horizontally combined with SciPy's `hstack`.

### Model training and selection

The data is split into training and test sets using an 80/20 split, `random_state=42`, and category stratification.

For each target (category, intent, priority), the training script compares:
- Logistic Regression
- Linear Support Vector Classifier (`LinearSVC`)

Both use class balancing. The script evaluates accuracy and macro F1, prints classification reports and confusion matrices, and selects the model with the higher macro F1 (accuracy is the tie-breaker).

The selected models and vectorizers are serialized with Joblib.

### Model artifacts

The Django classifier expects these files in `ml_training/`:

- `word_tfidf_vectorizer.pkl`
- `char_tfidf_vectorizer.pkl`
- `category_model.pkl`
- `intent_model.pkl`
- `priority_model.pkl`

The training script also writes `model_metrics.pkl`.

The archive includes additional model artifacts such as `final_hard_svm_model.pkl` and `final_hard_svm_tfidf.pkl`; the current `support_app/classifier.py` loads the five files listed above, not those additional files.

### Confidence calculation

`LinearSVC` does not provide calibrated class probabilities. The classifier therefore converts the decision-function margin into a bounded score:

```text
confidence = margin / (1 + margin)
```

For multiclass predictions, the margin is the difference between the highest and second-highest class scores. The overall confidence is a weighted combination:

```text
overall confidence =
    50% category confidence
  + 25% intent confidence
  + 25% priority confidence
```

This is a heuristic confidence score, **not a statistically calibrated probability**.

The classifier's initial ML action is:
- Confidence ≥ 0.80: `resolve`
- Confidence ≥ 0.50 and < 0.80: `draft`
- Confidence < 0.50: `escalate`

The decision engine subsequently applies a different policy, described below.

## Decision and automation logic

`support_app/engine.py` separates model prediction from the final operational decision.

### Rule matching

Active automation rules are checked in database order. A rule matches when:
- Its category equals the predicted category, or its category is blank.
- Its priority equals the predicted priority, or its priority is blank.

A blank field acts as a wildcard. The first matching active rule is used.

### Final action precedence

The decision engine follows this order:

1. **Matching active automation rule:** use the rule's configured action.
2. **Confidence is unavailable:** use `draft`.
3. **Confidence ≥ 0.90:** use the ML-predicted action.
4. **Confidence ≥ 0.70 and < 0.90:** preserve `escalate` if the ML action is escalate; otherwise use `draft`.
5. **Confidence < 0.70:** use `draft`.

This means the final decision can differ from the initial ML action. In particular, low-confidence model predictions are not automatically escalated by the final engine; absent a matching rule, they are sent for draft/review.

### Ticket status mapping

After the final action is selected:

| Final action | Ticket status |
|---|---|
| `resolve` | `resolved` |
| `draft` | `open` |
| `escalate` | `escalated` |

The engine saves the ticket, creates a decision-history record, and increments the trigger count for a matched automation rule.

## AI response generation

`support_app/response_generator.py` creates a customer-facing draft from the predicted category, intent, and priority.

The current implementation uses predefined text templates for:
- Refund requests and refund-status questions.
- Invoice queries.
- Billing and payment questions.
- Account/password requests.
- Account creation and registration.
- General or unrecognized requests.

This is template-based response generation, not a call to a hosted generative-AI API. The generated response is stored in `Ticket.ai_response`.

## Application pages and routes

The URL patterns are defined in `support_project/urls.py`.

| Route | Page / purpose |
|---|---|
| `/` | Dashboard |
| `/ticket_list/` | Ticket list, search, filters, and sorting |
| `/tickets/new/` | Create a ticket |
| `/tickets/generate/` | POST endpoint to generate a random dataset-based test ticket |
| `/tickets/<id>/` | Ticket detail and manual status update |
| `/automation-rules/` | Create, toggle, and delete automation rules |
| `/decision-log/` | Historical decision records and summary counts |
| `/settings/` | Organization, agent, inbox, notification, and account settings |
| `/admin/` | Django administration |

The main templates are:
- `dashboard.html`
- `ticket_list.html`
- `create_ticket.html`
- `ticket_detail.html`
- `automation_rules.html`
- `decision_log.html`
- `settings.html`

## Data models

All application models are defined in `support_app/models.py`.

### `Ticket`
Stores the customer request and its processing outcome:
- Title, customer name, message, and creation timestamp.
- Predicted category, intent, and priority.
- Current status.
- ML-predicted action and confidence.
- Final action and decision reason.
- Optional foreign key to the automation rule that was applied.
- Human-override flag.
- Generated AI/template response.

### `AutomationRule`
Stores configurable decision guardrails:
- Rule name.
- Optional category and priority match conditions.
- Action (`resolve`, `draft`, or `escalate`).
- Active/inactive state.
- Trigger count and creation timestamp.

### `DecisionLog`
Stores a historical snapshot of a ticket decision, including the ticket, classification, priority, ML action, confidence, matched rule, final action, reason, and timestamp. Records are ordered newest first.

### `OrganizationSettings`
Stores organization name, support email, timezone, agent active flag, and business-hour start/end values.

### `WatchedInbox`
Stores an inbox label, email address, active flag, and connection timestamp. In the supplied code this is configuration data; there is no implemented mailbox polling or email ingestion service.

### `NotificationPreference`
Stores preferences for escalation emails, new-ticket emails, and daily summary emails, plus a summary send time. The current project does not include the background task or email-sending implementation that would deliver these notifications.

## Project structure

```text
Automation/
└── support_project/
    ├── manage.py
    ├── db.sqlite3
    ├── support_project/
    │   ├── __init__.py
    │   ├── settings.py
    │   ├── urls.py
    │   ├── asgi.py
    │   └── wsgi.py
    ├── support_app/
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── apps.py
    │   ├── classifier.py
    │   ├── engine.py
    │   ├── models.py
    │   ├── response_generator.py
    │   ├── views.py
    │   ├── tests.py
    │   └── migrations/
    ├── templates/
    │   ├── dashboard.html
    │   ├── ticket_list.html
    │   ├── create_ticket.html
    │   ├── ticket_detail.html
    │   ├── automation_rules.html
    │   ├── decision_log.html
    │   └── settings.html
    ├── static/
    │   └── ticket_list.css
    └── ml_training/
        ├── dataset.csv
        ├── cleaned_dataset.csv
        ├── clean_dataset.py
        ├── intent_mapping.py
        ├── train_model.py
        ├── check_train_test_leakage.py
        ├── train_test_leakage_results.csv
        ├── model_metrics.pkl
        └── *.pkl                 # trained models and vectorizers
```

The archive also contains a `.vscode/settings.json` file and Python bytecode cache directories. Cache files are not required to run the project.

### Important source files

| File | Responsibility |
|---|---|
| `support_app/views.py` | Handles dashboard, ticket, automation-rule, decision-log, settings, and test-ticket requests |
| `support_app/classifier.py` | Loads trained artifacts, vectorizes a message, predicts labels, and calculates confidence |
| `support_app/engine.py` | Matches automation rules, chooses final action, updates ticket status, and writes decision history |
| `support_app/response_generator.py` | Produces template-based customer response text |
| `support_app/models.py` | Defines database entities |
| `support_app/admin.py` | Registers application models in Django admin |
| `support_project/settings.py` | Django configuration and SQLite/static-file settings |
| `support_project/urls.py` | Maps URL routes to views |
| `ml_training/clean_dataset.py` | Prepares the training dataset |
| `ml_training/intent_mapping.py` | Maps categories and intents to priority labels |
| `ml_training/train_model.py` | Trains, evaluates, selects, and saves models |
| `ml_training/check_train_test_leakage.py` | Measures nearest train/test message similarity |

## Technology stack

- **Python** — application and ML code.
- **Django** — web framework, routing, templates, ORM, admin, sessions, and messages.
- **SQLite** — default relational database.
- **Pandas** — dataset loading and cleaning.
- **scikit-learn** — TF-IDF vectorization, Logistic Regression, Linear SVM, train/test split, and evaluation.
- **HTML/CSS** — server-rendered interface and styling.

## Requirements

The supplied archive does not include a `requirements.txt` or other dependency lock file. The source imports the following main Python packages:

- Django
- pandas
- scikit-learn

Use a Python version supported by the Django and scikit-learn versions you install. The project settings header indicates it was generated with Django 6.0.7, but the archive does not pin that version. Install compatible, tested versions in a virtual environment.

## Setup and run locally

The Django project root is the `support_project/` directory inside the extracted `Automation/` folder.

### 1. Extract the archive

Extract `Automation.zip`. Open a terminal in:

```text
Automation/support_project/
```

This directory should contain `manage.py`.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install django pandas scikit-learn scipy joblib
```

For repeatable installations, record the versions that work in your environment in a `requirements.txt` file.

### 4. Check model files

Before starting the Django development server, confirm that these files exist in `ml_training/`:

```text
word_tfidf_vectorizer.pkl
char_tfidf_vectorizer.pkl
category_model.pkl
intent_model.pkl
priority_model.pkl
```

`support_app/classifier.py` loads these files when the application imports the classifier. If any are missing, ticket processing will fail with a `FileNotFoundError`.

If the model artifacts are missing or need to be rebuilt, follow [Train or rebuild the models](#train-or-rebuild-the-models).

### 5. Apply database migrations

```bash
python manage.py migrate
```

### 6. Create an administrator account (optional)

```bash
python manage.py createsuperuser
```

### 7. Start the development server

```bash
python manage.py runserver
```

Open the local address printed by Django, normally:

```text
http://127.0.0.1:8000/
```

The admin interface is available at `/admin/` after logging in with a staff/superuser account.

## Train or rebuild the models

Run these commands from the `ml_training/` directory, because the dataset scripts use relative paths.

### 1. Prepare the dataset

Ensure `dataset.csv` is present, then run:

```bash
python clean_dataset.py
```

This creates or overwrites `cleaned_dataset.csv`.

### 2. Train the models

```bash
python train_model.py
```

The script trains category, intent, and priority classifiers, compares Logistic Regression with Linear SVM using macro F1 and accuracy, and writes the selected models and vectorizers to the same directory.

### 3. Restart Django

After replacing model artifacts, restart the Django development server so the classifier module loads the updated files in a fresh process.

### 4. Optional: inspect train/test similarity

```bash
python check_train_test_leakage.py
```

This script compares test messages with training messages using TF-IDF cosine similarity and writes `train_test_leakage_results.csv`.

**Note:** The leakage-check script uses an intent-stratified split, while `train_model.py` uses category stratification. Its split is therefore not exactly the same as the training script's split, despite the comment in the leakage-check script.

## Database and Django administration

The project uses SQLite by default. Its database path is configured in `support_project/settings.py` and points to `db.sqlite3` in the Django project root.

Migrations are stored in `support_app/migrations/`. Apply them with:

```bash
python manage.py migrate
```

The following models are registered in Django admin:
- AutomationRule
- Ticket
- DecisionLog
- OrganizationSettings
- WatchedInbox
- NotificationPreference

The application also exposes several management actions through its own settings and automation-rule pages.

## Configuration

Most environment-specific configuration is in `support_project/settings.py`:

- `SECRET_KEY`
- `DEBUG`
- `ALLOWED_HOSTS`
- `DATABASES`
- `TIME_ZONE`
- `STATIC_URL`
- `STATICFILES_DIRS`
- `STATIC_ROOT`

Before deployment:
- Move `SECRET_KEY` to an environment variable and replace the committed development key.
- Set `DEBUG = False`.
- Configure `ALLOWED_HOSTS`.
- Configure HTTPS, secure cookies, CSRF trusted origins, and production database settings as appropriate.
- Run `python manage.py collectstatic` as part of deployment.
- Use a production WSGI/ASGI server rather than Django's development server.

## Current limitations and production considerations

1. **No dependency lock file:** The archive has no `requirements.txt` or `pyproject.toml`. Dependency versions should be pinned and tested.
2. **Development security settings:** `settings.py` contains a hard-coded insecure development secret key, `DEBUG = True`, and an empty `ALLOWED_HOSTS`. These settings are not production-ready. Rotate the key if this project has been shared publicly.
3. **Model artifacts are required at import time:** The classifier loads the `.pkl` files as soon as it is imported. Missing or incompatible artifacts can prevent ticket-processing views from loading.
4. **Pickle/joblib trust:** Only load model files from trusted sources. Deserializing untrusted pickle/joblib files can execute arbitrary code.
5. **No live email ingestion:** `WatchedInbox` stores inbox metadata only. No IMAP/POP/Gmail/Outlook connection, polling task, or incoming-email-to-ticket workflow is implemented in the supplied source.
6. **No notification delivery:** Notification preferences are stored, but no email sender, scheduled daily summary, or background notification task is included.
7. **Agent switch is configuration only:** The settings view can toggle `OrganizationSettings.agent_is_active`, but the ticket-processing path shown does not check this flag before running the classifier/engine.
8. **Business hours are stored, not enforced:** Business-hour values and timezone are saved, but the current decision engine does not use them to defer or schedule processing.
9. **Response generation is template-based:** `generate_ai_response()` uses predefined strings; it does not invoke a generative AI model or external API.
10. **Confidence is heuristic:** The SVM margin conversion is not calibrated probability. Thresholds should be validated against representative, held-out production data before enabling automatic resolution.
11. **Training split and evaluation:** Training uses category stratification for all three targets. Review class counts and per-class metrics, especially for rare intents, before relying on the reported overall metrics.
12. **No meaningful automated test suite is included:** `support_app/tests.py` currently contains only the default empty test class placeholder. Add tests for classifier outputs, rule precedence, confidence thresholds, status mapping, and ticket creation.
13. **Manual overrides are limited:** The ticket detail view updates status and sets `was_overridden_by_human`; it does not create a separate audit event describing the old and new status.
14. **Potential rule-order ambiguity:** The engine uses the first matching active rule. Consider explicit rule priority/order and conflict handling if multiple rules can match the same ticket.
15. **No authentication decorators on application views:** The supplied URL/view code does not visibly restrict dashboard, ticket, settings, or rule-management views to authenticated staff. Add authorization and role checks before exposing the application.
16. **No transaction boundary around processing:** Ticket updates, decision-log creation, and rule-trigger increments are separate database operations. Consider `transaction.atomic()` to keep them consistent if one step fails.
17. **Input validation should be strengthened:** Validate required ticket fields and allowed values server-side; do not rely only on template-level form validation.
18. **SQLite is suitable for development/small deployments:** Evaluate PostgreSQL or another production database for concurrent users and operational workloads.

## Troubleshooting

- **`FileNotFoundError` for a `.pkl` file:** Confirm all five classifier artifacts are present in `ml_training/`. Re-run the training script if needed.
- **`ModuleNotFoundError`:** Activate the intended virtual environment and install the missing package.
- **`no such table` database error:** Run `python manage.py migrate` from the directory containing `manage.py`.
- **Template not found:** Confirm the `templates/` directory is alongside `manage.py`, as configured by `TEMPLATES['DIRS']`.
- **Static CSS is missing:** Confirm `static/ticket_list.css` exists and check the template's static-file references. For deployment, configure and collect static files.
- **Model loading or version errors:** Reinstall compatible versions of scikit-learn/joblib or retrain the model artifacts in the same environment used by the application.
- **Ticket processing fails after training:** Check that the vectorizers and all three target models were generated and saved to the exact filenames expected by `classifier.py`.
- **Test-ticket generation fails:** Confirm `ml_training/cleaned_dataset.csv` exists and includes a non-empty `message` column.

## Disclaimer

This project is a development example of machine-learning-assisted support-ticket triage and rule-based automation. Model predictions and generated response text may be incorrect. Do not enable automatic customer-facing actions in a production support environment without appropriate authentication, authorization, monitoring, human review, testing, and operational safeguards.
