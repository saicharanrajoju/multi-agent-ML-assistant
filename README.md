# Multi-Agent ML Assistant: Autonomous Machine Learning Pipeline

## Overview

**Multi-Agent ML Assistant** is a sophisticated, human-in-the-loop multi-agent system designed to automate the end-to-end machine learning lifecycle. Built on **LangGraph**, it orchestrates a team of specialized AI agents that collaboratively profile data, clean datasets, engineer features, train models, and deploy production-ready inference APIs.

This system emphasizes **reliability, transparency, and control**, integrating **E2B Sandboxes** for secure code execution and providing interactive breakpoints for human oversight. It transforms raw CSV data into deployed ML solutions with minimal human intervention while maintaining high standards of code quality and model performance.

## Key Features

- **🤖 Autonomous Multi-Agent Architecture**: A team of specialized agents (Profiler, Cleaner, Feature Engineer, Modeler, Critic, Deployer) working in concert.
- **🔄 State Machine Orchestration**: Leveraging **LangGraph** to manage complex, cyclic workflows with state persistence and error recovery.
- **🛡️ Secure Code Execution**: All generated Python code runs inside isolated **E2B Sandboxes**, ensuring safety and reproducibility.
- **👨‍💻 Human-in-the-Loop**: Strategic interrupt points allow users to review, approve, or provide feedback on agent outputs (e.g., cleaning logic, feature selection) before execution.
- **📈 Self-Correcting Mechanisms**: A dedicated **Critic Agent** evaluates model performance and loops back to feature engineering if metrics fall short of goals.
- **🚀 Automated Deployment**: Generates a production-ready **FastAPI** application Dockerized for immediate deployment.

## System Architecture

The pipeline follows a directed cyclic graph (DAG) structure:

1.  **Profiler Agent**: Analyzes the raw dataset to understand distribution, missing values, and data types.
2.  **Cleaner Agent**: Generates and executes Python code to handle missing values, duplicates, and outliers.
3.  **Feature Engineer Agent**: Creates new features and transforms data to maximize model performance.
4.  **Modeler Agent**: Selects algorithms, trains models, and tunes hyperparameters.
5.  **Critic Agent**: Evaluates the model against the user's goal. If unsatisfactory, it provides feedback and routes control back to Feature Engineering.
6.  **Deployer Agent**: packages the best model into a FastAPI service.

## Tech Stack

-   **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/), [LangChain](https://www.langchain.com/)
-   **LLM Integration**: [Groq](https://groq.com/) (Llama 3 via LangChain)
-   **Sandboxing**: [E2B](https://e2b.dev/)
-   **Data Science**: Pandas, Scikit-learn, NumPy
-   **Visualization**: Plotly
-   **Deployment**: Docker, FastAPI

## Getting Started

### Prerequisites

-   Python 3.10+
-   Docker (for deployment)
-   API Keys for **Groq**, **E2B**, **LangSmith** (optional)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/saicharanrajoju/multi-agent-ML-assistant.git
    cd multi-agent-ML-assistant
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_api_key
    E2B_API_KEY=your_e2b_api_key
    LANGCHAIN_API_KEY=your_langchain_api_key
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=ml-agent-assistant
    ```

### Usage

Run the pipeline by specifying your dataset and objective:

```bash
python main.py "datasets/your_dataset.csv" "Predict [target_column] with [metric_goal]"
```

**Example:**
```bash
python main.py "datasets/customer_churn.csv" "Predict Churn with F1-score > 0.8"
```

### The Workflow

1.  The system will print the initial data profile.
2.  **Review Point**: You will be prompted to approve the data cleaning plan.
    -   Type `a` to approve.
    -   Type `f` to provide feedback (e.g., "Don't drop column X").
3.  The pipeline continues through feature engineering and modeling.
4.  If the **Critic** is not satisfied, it will automatically iterate.
5.  Upon success, the **Deployer** generates a `deployment/` folder with a Dockerfile and `app.py`.

### Deployment

To serve the generated model:

```bash
cd outputs/deployment
docker-compose up --build
```

The API will be available at `http://localhost:8000/predict`.

## Project Structure

```
├── main.py                 # Entry point
├── src/
│   ├── agents/             # Agent definitions (Cleaner, Modeler, etc.)
│   ├── tools/              # Tools (Sandbox execution, File I/O)
│   ├── graph.py            # LangGraph workflow definition
│   └── state.py            # Global state schema
├── datasets/               # Input data storage
├── outputs/                # Generated artifacts (reports, models, code)
├── requirements.txt        # Python dependencies
└── docker-compose.yml      # Deployment orchestration
```

## Contributing

Contributions are welcome. Please ensure all modifications include appropriate tests and documentation.

## License

[MIT License](LICENSE)
