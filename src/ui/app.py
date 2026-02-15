import streamlit as st
import os
import sys
import time
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.figure_factory as ff
import contextlib
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.graph import graph
from src.state import AgentState
from src.tools.code_executor import SandboxManager

# Page config
st.set_page_config(
    page_title="Multi-Agent ML Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "pipeline_config" not in st.session_state:
    st.session_state.pipeline_config = None
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "pipeline_complete" not in st.session_state:
    st.session_state.pipeline_complete = False
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "logs" not in st.session_state:
    st.session_state.logs = []
if "waiting_for_approval" not in st.session_state:
    st.session_state.waiting_for_approval = False
if "next_node" not in st.session_state:
    st.session_state.next_node = None
if "full_debug_log" not in st.session_state:
    st.session_state.full_debug_log = "--- System Log init ---\n"

AGENT_STEPS = [
    {"name": "profiler", "icon": "🔍", "label": "Data Profiling"},
    {"name": "cleaner", "icon": "🧹", "label": "Data Cleaning"},
    {"name": "feature_engineer", "icon": "⚙️", "label": "Feature Engineering"},
    {"name": "modeler", "icon": "🤖", "label": "Model Training"},
    {"name": "critic", "icon": "🧐", "label": "Pipeline Critique"},
    {"name": "deployer", "icon": "🚀", "label": "Deployment"},
]

# --- SIDEBAR ---
with st.sidebar:
    st.title("🤖 ML Agent Assistant")
    st.markdown("---")

    # Sidebar to reset history
    st.header("⚙️ Configuration")
    if st.button("Reset Agent Memory"):
        if "thread_id" in st.session_state:
            del st.session_state["thread_id"]
        if "messages" in st.session_state:
            del st.session_state["messages"]
        st.rerun()

    st.markdown("---")
    st.markdown("### 📂 Choose Dataset")
    
    # Scan datasets folder for available CSVs
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    available_datasets = []
    if os.path.exists(datasets_dir):
        for f in os.listdir(datasets_dir):
            if f.endswith('.csv'):
                available_datasets.append(f)
    
    dataset_choice = st.radio(
        "Select a dataset:",
        options=["Upload my own"] + available_datasets,
        index=0,
    )

    dataset_path = ""
    if dataset_choice == "Upload my own":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            # Save uploaded file
            save_path = os.path.join(datasets_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            dataset_path = uploaded_file.name
            st.success(f"Uploaded: {uploaded_file.name}")
    else:
        dataset_path = dataset_choice

    # Sidebar ends here
    
# --- MAIN AREA ---
st.title("🤖 Multi-Agent ML Assistant")
st.markdown("*An AI-powered pipeline that automates ML workflows with human oversight*")

# Smart goal suggestions based on selected dataset
goal_suggestions = {
    "WA_Fn-UseC_-Telco-Customer-Churn.csv": "Predict customer churn with high F1-score and low false positive rate",
    "titanic.csv": "Predict passenger survival with high recall to identify all survivors",
}

default_goal = ""
if "dataset_choice" in locals() and dataset_choice in goal_suggestions:
    default_goal = goal_suggestions[dataset_choice]

# User Input
if "dataset_path" in locals() and dataset_path:
    st.info(f"Using dataset: `{dataset_path}`")
else:
    st.warning("Please select or upload a dataset in the sidebar.")

user_goal = st.text_area(
    "🎯 Modeling Goal",
    value=default_goal,
    placeholder="e.g., Predict customer churn with high accuracy...",
    height=100
)

# Run button
if st.button("🚀 Run Pipeline", type="primary", use_container_width=True, disabled=st.session_state.pipeline_running):
    if not user_goal:
        st.error("Please enter a goal.")
    elif not ("dataset_path" in locals() and dataset_path):
        st.error("Please select a dataset.")
    else:
        st.session_state.pipeline_running = True
        st.session_state.pipeline_complete = False
        st.session_state.current_step = 0
        st.session_state.logs = []
        st.session_state.waiting_for_approval = False

        # Create config with unique thread ID
        thread_id = f"streamlit-run-{int(time.time())}"
        st.session_state.pipeline_config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "dataset_path": dataset_path,
            "user_goal": user_goal,
            "messages": [],
            "iteration_count": 0,
            "should_iterate": False,
            "cleaning_approved": False,
            "feature_approved": False,
            "model_approved": False,
            "deployment_approved": False,
            "human_feedback": "",
            "error": "",
            "visualization_data": {},
            "scorecard": {},
            "code_fixes": [],
            "iteration_history": [],
            "dataset_summary": {},
            "cleaning_summary": {},
            "target_column": "",
        }
        st.session_state.pipeline_state = initial_state

        # Start the pipeline
        start_msg = f"[{time.strftime('%H:%M:%S')}] 🚀 Pipeline started!\n"
        st.session_state.logs.append("🚀 Pipeline started!")
        st.session_state.full_debug_log += start_msg
        
        try:
            # Capture stdout to log agent print statements
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                for event in graph.stream(initial_state, st.session_state.pipeline_config, stream_mode="values"):
                    agent = event.get("current_agent", "")
                    if agent:
                        msg = f"✅ {agent.upper()} completed"
                        st.session_state.logs.append(msg)
                        st.session_state.pipeline_state = event
            
            # Append captured stdout to full log
            captured_log = stdout_capture.getvalue()
            st.session_state.full_debug_log += captured_log
            
            # Optional: dump final state keys for structure check
            # st.session_state.full_debug_log += f"    State keys update: {list(event.keys())}\n"

            # Check if waiting at interrupt
            state = graph.get_state(st.session_state.pipeline_config)
            if state.next:
                st.session_state.waiting_for_approval = True
                st.session_state.next_node = state.next[0]
                msg = f"⏸️ Waiting for approval before: {state.next[0]}"
                st.session_state.logs.append(msg)
                st.session_state.full_debug_log += f"[{time.strftime('%H:%M:%S')}] {msg}\n"
            else:
                st.session_state.pipeline_complete = True
                st.session_state.pipeline_running = False
                end_msg = f"[{time.strftime('%H:%M:%S')}] 🎉 Pipeline complete!\n"
                st.session_state.full_debug_log += end_msg
                SandboxManager.reset()
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.logs.append(error_msg)
            st.session_state.pipeline_running = False
            
            # Full traceback for debug log
            tb = traceback.format_exc()
            st.session_state.full_debug_log += f"\n[{time.strftime('%H:%M:%S')}] ❌ EXCEPTION OCCURRED:\n{tb}\n"
            
            SandboxManager.reset()

        st.rerun()

# Reset button
if st.button("🔄 Reset Logic", use_container_width=True):
    SandboxManager.reset()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
    
# Create tabs
tab_progress, tab_profile, tab_results, tab_critique, tab_deployment = st.tabs([
    "📋 Pipeline Progress",
    "📊 Data Profile",
    "📈 Model Results",
    "🧐 Critique & Scorecard",
    "🚀 Deployment",
])

with tab_progress:
    # Show logs
    if st.session_state.logs:
        st.subheader("Pipeline Log")
        log_container = st.container()
        with log_container:
            for log in st.session_state.logs:
                if "Error" in log or "❌" in log:
                    st.error(log)
                elif "✅" in log:
                    st.success(log)
                elif "⏸️" in log:
                    st.warning(log)
                else:
                    st.info(log)

    # Human review section
    if st.session_state.waiting_for_approval:
        st.markdown("---")
        st.subheader(f"⏸️ Review Required: {st.session_state.next_node}")

        state_vals = st.session_state.pipeline_state or {}

        # Show relevant info
        if st.session_state.next_node == "cleaner":
            st.markdown("### Data Profile Summary")
            st.markdown(state_vals.get("profile_report", "No report yet")[:2000])
            issues = state_vals.get("data_issues", [])
            if issues:
                st.markdown("### Issues Found")
                for issue in issues:
                    st.markdown(f"- {issue}")

        elif st.session_state.next_node == "feature_engineer":
            st.markdown("### Cleaning Results")
            st.text(state_vals.get("cleaning_result", "No result yet"))
            with st.expander("View Cleaning Code"):
                st.code(state_vals.get("cleaning_code", "No code yet"), language="python")

        elif st.session_state.next_node == "modeler":
            st.markdown("### Feature Engineering Results")
            st.text(state_vals.get("feature_result", "No result yet"))
            with st.expander("View Feature Engineering Code"):
                st.code(state_vals.get("feature_code", "No code yet"), language="python")

        elif st.session_state.next_node == "deployer":
            st.markdown("### Model Training Results")
            st.text(state_vals.get("model_result", "No result yet"))
            st.markdown("### Critic Feedback")
            st.markdown(state_vals.get("critique_report", "No critique yet")[:1500])

        # Approval buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Approve & Continue", type="primary", use_container_width=True):
                st.session_state.waiting_for_approval = False
                st.session_state.logs.append(f"✅ Approved: {st.session_state.next_node}")

                try:
                    # Capture stdout to log agent print statements
                    stdout_capture = io.StringIO()
                    with contextlib.redirect_stdout(stdout_capture):
                        for event in graph.stream(None, st.session_state.pipeline_config, stream_mode="values"):
                            agent = event.get("current_agent", "")
                            if agent:
                                msg = f"✅ {agent.upper()} completed"
                                st.session_state.logs.append(msg)
                                st.session_state.pipeline_state = event
                    
                    # Append captured stdout to full log
                    captured_log = stdout_capture.getvalue()
                    st.session_state.full_debug_log += captured_log

                    # Check for next interrupt
                    state = graph.get_state(st.session_state.pipeline_config)
                    if state.next:
                        st.session_state.waiting_for_approval = True
                        st.session_state.next_node = state.next[0]
                        st.session_state.logs.append(f"⏸️ Waiting for approval before: {state.next[0]}")
                    else:
                        st.session_state.pipeline_complete = True
                        st.session_state.pipeline_running = False
                        st.session_state.logs.append("🎉 Pipeline complete!")
                        SandboxManager.reset()
                except Exception as e:
                    st.session_state.logs.append(f"❌ Error: {str(e)}")
                    st.session_state.pipeline_running = False
                    SandboxManager.reset()

                st.rerun()

        with col2:
            feedback = st.text_input("📝 Feedback (optional)")
            if st.button("📝 Submit Feedback & Continue", use_container_width=True):
                st.session_state.waiting_for_approval = False
                if feedback:
                    graph.update_state(st.session_state.pipeline_config, {"human_feedback": feedback})
                    st.session_state.logs.append(f"📝 Feedback submitted: {feedback}")

                try:
                    # Capture stdout to log agent print statements
                    stdout_capture = io.StringIO()
                    with contextlib.redirect_stdout(stdout_capture):
                        for event in graph.stream(None, st.session_state.pipeline_config, stream_mode="values"):
                            agent = event.get("current_agent", "")
                            if agent:
                                msg = f"✅ {agent.upper()} completed"
                                st.session_state.logs.append(msg)
                                st.session_state.pipeline_state = event

                    # Append captured stdout to full log
                    captured_log = stdout_capture.getvalue()
                    st.session_state.full_debug_log += captured_log

                    state = graph.get_state(st.session_state.pipeline_config)
                    if state.next:
                        st.session_state.waiting_for_approval = True
                        st.session_state.next_node = state.next[0]
                        st.session_state.logs.append(f"⏸️ Waiting for approval before: {state.next[0]}")
                    else:
                        st.session_state.pipeline_complete = True
                        st.session_state.pipeline_running = False
                        st.session_state.logs.append("🎉 Pipeline complete!")
                        SandboxManager.reset()
                except Exception as e:
                    st.session_state.logs.append(f"❌ Error: {str(e)}")
                    st.session_state.pipeline_running = False
                    SandboxManager.reset()

                st.rerun()

        with col3:
            if st.button("🛑 Stop Pipeline", use_container_width=True):
                st.session_state.pipeline_running = False
                st.session_state.waiting_for_approval = False
                st.session_state.logs.append("🛑 Pipeline stopped by user")
                SandboxManager.reset()
                st.rerun()

    elif st.session_state.pipeline_complete:
        st.success("🎉 Pipeline completed successfully!")
        st.balloons()

    elif not st.session_state.pipeline_running and not st.session_state.logs:
        st.info("👈 Upload a dataset and click 'Run Pipeline' to get started!")


with tab_profile:
    state_vals = st.session_state.pipeline_state or {}
    report = state_vals.get("profile_report", "")
    if report:
        st.markdown("## Data Profile Report")
        st.markdown(report)

        issues = state_vals.get("data_issues", [])
        if issues:
            st.markdown("---")
            st.markdown("## Identified Issues")
            for issue in issues:
                st.warning(f"⚠️ {issue}")
    else:
        st.info("Data profile will appear here after the Profiler Agent runs.")


with tab_results:
    state_vals = st.session_state.pipeline_state or {}
    viz_data = state_vals.get("visualization_data", {})
    
    if viz_data:
        st.markdown("## 📈 Model Performance Dashboard")
        
        # Top level metrics
        if "best_model" in viz_data:
            bm = viz_data["best_model"]
            reports = bm.get("classification_report", {})
            
            # Extract weighted avg or macro avg if available, or class 1 for churn
            f1 = reports.get("f1_1", 0)  # assume binary classification class 1
            acc = reports.get("accuracy", 0)
            
            # Find best model metrics from comparison
            best_model_name = bm.get("name", "Unknown")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏆 Best Model", best_model_name)
            
            # Try to get data from comparison to be precise
            if "model_comparison" in viz_data:
                comp = viz_data["model_comparison"]
                try:
                    idx = comp["model_names"].index(best_model_name)
                    col2.metric("F1 Score", f"{comp['f1_score'][idx]:.3f}")
                    col3.metric("AUC-ROC", f"{comp['auc_roc'][idx]:.3f}")
                    col4.metric("Accuracy", f"{comp['accuracy'][idx]:.3f}")
                except:
                    col2.metric("F1 Score", "N/A")

        st.markdown("---")
        
        # A) Model Comparison
        if "model_comparison" in viz_data:
            st.subheader("📊 Model Comparison")
            comparison = viz_data["model_comparison"]
            fig = go.Figure()
            metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
            
            for metric, color in zip(metrics, colors):
                if metric in comparison:
                    fig.add_trace(go.Bar(
                        name=metric.replace("_", " ").title(),
                        x=comparison["model_names"],
                        y=comparison[metric],
                        marker_color=color,
                    ))
            
            fig.update_layout(
                barmode='group',
                title="Model Performance Metrics by Model",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                template="plotly_white",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        col_left, col_right = st.columns(2)
        
        with col_left:
            # B) Confusion Matrix
            if "best_model" in viz_data:
                cm = viz_data["best_model"].get("confusion_matrix")
                if cm:
                    st.subheader(f"🟦 Confusion Matrix: {viz_data['best_model']['name']}")
                    labels = ["No Churn", "Churn"] # This might need to be dynamic based on dataset
                    
                    # Flatten logic just in case, or handle list of lists
                    z = cm
                    x = labels
                    y = labels
                    
                    # Invert Y for heatmap to match standard confusion matrix layout if needed, 
                    # but plotly heatmap usually starts bottom-left. 
                    # Standard CM: True Label on Y (Top-Down), Predicted on X (Left-Right)
                    
                    fig_cm = ff.create_annotated_heatmap(
                        z=z,
                        x=x,
                        y=y,
                        colorscale="Blues",
                        showscale=True,
                    )
                    fig_cm.update_layout(
                        xaxis_title="Predicted Label",
                        yaxis_title="True Label",
                        height=400,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

        with col_right:
            # D) Cross-Validation
            if "cross_validation" in viz_data:
                cv = viz_data["cross_validation"]
                st.subheader("📉 Cross-Validation Consistency")
                
                scores = cv.get("cv_scores", [])
                if scores:
                    mean_val = cv.get("mean", sum(scores)/len(scores))
                    std_val = cv.get("std", 0)
                    
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Bar(
                        x=[f"Fold {i+1}" for i in range(len(scores))],
                        y=scores,
                        marker_color="#00CC96",
                        name="Fold Score"
                    ))
                    fig_cv.add_hline(
                        y=mean_val, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Mean: {mean_val:.3f} ± {std_val:.3f}",
                        annotation_position="top right"
                    )
                    fig_cv.update_layout(
                        yaxis_title="Score",
                        yaxis=dict(range=[max(0, mean_val-0.2), min(1, mean_val+0.2)]), # Zoom in a bit
                        template="plotly_white",
                        height=400,
                    )
                    st.plotly_chart(fig_cv, use_container_width=True)

        # C) Feature Importance
        if "best_model" in viz_data:
            fi = viz_data["best_model"].get("feature_importance")
            if fi:
                st.subheader("✨ Top Feature Importances")
                
                names = fi["feature_names"][:15]
                values = fi["importance_values"][:15]
                
                # Sort for chart (ascending for horizontal bar to show top at top)
                names_sorted = names[::-1]
                values_sorted = values[::-1]
                
                fig_fi = go.Figure(go.Bar(
                    x=values_sorted,
                    y=names_sorted,
                    orientation='h',
                    marker_color="#636EFA",
                ))
                fig_fi.update_layout(
                    xaxis_title="Importance Score",
                    template="plotly_white",
                    height=600,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        # Raw results fallback
        with st.expander("View Raw Training Results & Code"):
            st.text(state_vals.get("model_result", ""))
            st.code(state_vals.get("model_code", "No code"), language="python")
            
    else:
        # Fallback if no viz data yet
        model_result = state_vals.get("model_result", "")
        if model_result:
            st.markdown("## Model Training Results")
            st.text(model_result)
            with st.expander("View Model Training Code"):
                st.code(state_vals.get("model_code", "No code"), language="python")
        else:
            st.info("Model results and visualizations will appear here after the Modeler Agent runs.")


with tab_critique:
    state_vals = st.session_state.pipeline_state or {}
    critique = state_vals.get("critique_report", "")
    scorecard = state_vals.get("scorecard", {})
    
    if critique or scorecard:
        st.markdown("## 🧐 Critique & Scorecard")
        
        # F) Scorecard Radar Chart
        if scorecard:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Pipeline Quality Scorecard")
                categories = list(scorecard.keys())
                if "overall" in categories:
                    categories.remove("overall")
                
                # Standard order if possible
                desired_order = ["data_leakage", "code_quality", "metric_alignment", 
                                 "feature_engineering", "model_selection", "deployment_readiness"]
                
                ordered_cats = [c for c in desired_order if c in categories]
                # Add any others found
                for c in categories:
                    if c not in ordered_cats:
                        ordered_cats.append(c)
                
                values = [scorecard.get(c, 0) for c in ordered_cats]
                labels = [c.replace("_", " ").title() for c in ordered_cats]
                
                fig_radar = go.Figure(go.Scatterpolar(
                    r=values + [values[0]],  # close the polygon
                    theta=labels + [labels[0]],
                    fill='toself',
                    fillcolor='rgba(99, 110, 250, 0.2)',
                    line=dict(color='#636EFA', width=2),
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10])
                    ),
                    height=450,
                    margin=dict(l=40, r=40, t=20, b=20),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                st.subheader("Details")
                overall = scorecard.get("overall", "N/A")
                st.metric("⭐ Overall Score", f"{overall}/10")
                
                st.markdown("### Breakdown")
                for cat, val in zip(labels, values):
                    st.progress(val/10, text=f"{cat}: {val}/10")

        st.markdown("---")
        st.markdown("### 📝 Detailed Critique Report")
        st.markdown(critique)

        suggestions = state_vals.get("improvement_suggestions", [])
        if suggestions:
            st.markdown("---")
            st.markdown("## Improvement Suggestions")
            for i, s in enumerate(suggestions, 1):
                st.info(f"💡 {i}. {s}")

        st.markdown(f"**Iterations completed:** {state_vals.get('iteration_count', 0)}")
        
        # Iteration History
        history = state_vals.get("iteration_history", [])
        if history:
            st.markdown("---")
            st.subheader("📜 Iteration History")
            for entry in history:
                with st.expander(f"Iteration {entry['iteration']} — Severity: {entry.get('severity', 'N/A')}"):
                    st.markdown(f"**Fixes required:** {entry.get('n_fixes', 0)}")
                    if entry.get('suggestions'):
                        st.markdown("**Suggestions:**")
                        for s in entry['suggestions']:
                            st.markdown(f"- {s}")
                    if entry.get('scorecard'):
                        st.json(entry['scorecard'])
    else:
        st.info("Critique report and scorecard will appear here after the Critic Agent runs.")


with tab_deployment:
    state_vals = st.session_state.pipeline_state or {}
    deployment_code = state_vals.get("deployment_code", "")
    if deployment_code:
        st.markdown("## Deployment Package")
        st.success(f"🌐 API Endpoint: {state_vals.get('api_endpoint', 'http://localhost:8000')}")

        with st.expander("View Deployment Generator Code"):
            st.code(deployment_code, language="python")

        # Check if deployment files exist
        deploy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "deployment")
        if os.path.exists(deploy_dir):
            st.markdown("### Generated Files")
            for f in os.listdir(deploy_dir):
                filepath = os.path.join(deploy_dir, f)
                st.markdown(f"📄 `{f}` ({os.path.getsize(filepath)} bytes)")

            st.markdown("### Quick Start")
            st.code("cd outputs/deployment\ndocker-compose up --build\n# API available at http://localhost:8000", language="bash")
    else:
        st.info("Deployment package will appear here after the Deployer Agent runs.")


# --- DEBUG CONSOLE (Added per user request) ---
with st.expander("🛠️ System Log / Debug Console", expanded=False):
    st.info("Copy this log to share tracebacks and errors with the developer.")
    
    # Download button
    if "full_debug_log" in st.session_state:
        st.download_button(
            label="💾 Download Full Log",
            data=st.session_state.full_debug_log,
            file_name=f"debug_log_{int(time.time())}.txt",
            mime="text/plain"
        )
        
        # Display log with native copy button
        st.code(
            st.session_state.full_debug_log, 
            language="text"
        )

# Footer
st.markdown("---")
st.markdown("*Built with LangGraph, Groq, E2B, and LangSmith | DTSC 5082 Group 2*")
