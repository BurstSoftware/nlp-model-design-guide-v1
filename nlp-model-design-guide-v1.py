import streamlit as st

# Title
st.title("NLP Model Design with Streamlit")
st.write("Design your custom NLP model by following the steps below.")

# Step 1: Define the Objective
st.header("1. Define the Objective")
task = st.selectbox(
    "Select the task:",
    ["Text Generation", "Translation", "Code Completion", "Question Answering"]
)
model_arch = st.selectbox(
    "Select the model architecture:",
    ["Transformer", "GPT-style", "LLaMA-like", "BERT", "T5"]
)
scale = st.slider(
    "Select the scale (number of parameters in billions):",
    min_value=0.1, max_value=100.0, step=0.1, value=1.0
)

# Step 2: Data Collection & Preprocessing
st.header("2. Data Collection & Preprocessing")
data_sources = st.multiselect(
    "Select data sources:",
    ["Books", "Wikipedia", "Research Papers", "Code Repositories", "Web Crawls"]
)
preprocessing_steps = st.multiselect(
    "Select preprocessing steps:",
    ["Remove Duplicates", "Tokenization", "Normalization", "Train/Test/Validation Split"]
)

# Step 3: Model Architecture Selection & Implementation
st.header("3. Model Architecture Selection & Implementation")
num_layers = st.slider("Number of layers:", min_value=1, max_value=100, value=12)
num_heads = st.slider("Number of attention heads:", min_value=1, max_value=32, value=8)
hidden_units = st.slider("Hidden units:", min_value=128, max_value=8192, value=768)
context_length = st.slider("Context length (tokens):", min_value=512, max_value=8192, value=2048)

# Step 4: Training Strategy
st.header("4. Training Strategy")
hardware = st.selectbox(
    "Select hardware:",
    ["GPUs (A100)", "GPUs (H100)", "TPUs"]
)
optimizer = st.selectbox(
    "Select optimizer:",
    ["AdamW", "Lion", "LAMB"]
)
scheduler = st.selectbox(
    "Select learning rate scheduler:",
    ["Cosine Decay", "Linear Decay"]
)

# Step 5: Deployment
st.header("5. Deployment")
quantization = st.checkbox("Enable Quantization (e.g., GPTQ, AWQ)")
inference_framework = st.selectbox(
    "Select inference framework:",
    ["TensorRT", "vLLM", "FasterTransformer"]
)
serving_framework = st.selectbox(
    "Select serving framework:",
    ["Triton Inference Server", "FastAPI", "Ray Serve"]
)

# Display Summary
st.header("Model Design Summary")
st.write(f"**Task:** {task}")
st.write(f"**Model Architecture:** {model_arch}")
st.write(f"**Scale:** {scale} billion parameters")
st.write(f"**Data Sources:** {', '.join(data_sources)}")
st.write(f"**Preprocessing Steps:** {', '.join(preprocessing_steps)}")
st.write(f"**Model Layers:** {num_layers}, **Heads:** {num_heads}, **Hidden Units:** {hidden_units}")
st.write(f"**Context Length:** {context_length} tokens")
st.write(f"**Hardware:** {hardware}")
st.write(f"**Optimizer:** {optimizer}")
st.write(f"**Scheduler:** {scheduler}")
st.write(f"**Quantization:** {'Enabled' if quantization else 'Disabled'}")
st.write(f"**Inference Framework:** {inference_framework}")
st.write(f"**Serving Framework:** {serving_framework}")

# Generate Configuration File (Optional)
if st.button("Generate Configuration File"):
    config = {
        "task": task,
        "model_architecture": model_arch,
        "scale": scale,
        "data_sources": data_sources,
        "preprocessing_steps": preprocessing_steps,
        "model_layers": num_layers,
        "model_heads": num_heads,
        "hidden_units": hidden_units,
        "context_length": context_length,
        "hardware": hardware,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "quantization": quantization,
        "inference_framework": inference_framework,
        "serving_framework": serving_framework,
    }
    st.download_button(
        label="Download Config",
        data=str(config),
        file_name="model_config.json",
        mime="application/json"
    )

st.write("Thank you for designing your NLP model!")
