import streamlit as st
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector
from metrics import calculate_fidelity, trace_distnace, calculate_QJSD, is_pure, get_bloch_vector
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantum Similarity Dashboard", layout="wide")
st.title("🔬 Quantum State Similarity Tool")

def get_state(choice):
    if choice == "|0⟩":
        return Statevector([1, 0])
    elif choice == "|1⟩":
        return Statevector([0, 1])
    elif choice == "|+⟩":
        return Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
    elif choice == "|−⟩":
        return Statevector([1/np.sqrt(2), -1/np.sqrt(2)])
    elif choice == "Mixed (90% |0⟩ + 10% |1⟩)":
        return 0.9 * DensityMatrix([1, 0]) + 0.1 * DensityMatrix([0, 1])
    elif choice == "Mixed (50/50)":
        return 0.5 * DensityMatrix([1, 0]) + 0.5 * DensityMatrix([0, 1])


st.sidebar.header("Select Quantum States")
state1 = st.sidebar.selectbox("Choose State 1", ["|0⟩", "|1⟩", "|+⟩", "|−⟩", "Mixed (90% |0⟩ + 10% |1⟩)", "Mixed (50/50)"])
state2 = st.sidebar.selectbox("Choose State 2", ["|0⟩", "|1⟩", "|+⟩", "|−⟩", "Mixed (90% |0⟩ + 10% |1⟩)", "Mixed (50/50)"])

rho = get_state(state1)
sigma = get_state(state2)


if isinstance(rho, Statevector):
    rho_dm = DensityMatrix(rho)
else:
    rho_dm = rho

if isinstance(sigma, Statevector):
    sigma_dm = DensityMatrix(sigma)
else:
    sigma_dm = sigma

fidelity = calculate_fidelity(rho_dm, sigma_dm)
trace = trace_distnace(rho_dm, sigma_dm)
qjsd = calculate_QJSD(rho_dm, sigma_dm)
purity_rho = np.trace(rho_dm.data @ rho_dm.data)
purity_sigma = np.trace(sigma_dm.data @ sigma_dm.data)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Similarity Metrics")
    st.write(f"**Fidelity:** {fidelity:.3f}")
    st.write(f"**Trace Distance:** {trace:.3f}")
    st.write(f"**QJSD:** {qjsd:.3f}")

    st.subheader("🧼 Purity")
    st.write(f"Purity of ρ: {purity_rho:.3f}")
    st.write(f"Purity of σ: {purity_sigma:.3f}")

    if is_pure(rho) and is_pure(sigma):
        st.success("Both states are pure. Expect D = sqrt(1 - F).")
    else:
        st.warning("At least one state is mixed. D and F may not be strictly related.")

with col2:
    st.subheader("🔵 Bloch Sphere for Rho and Sigma")
    st.write("State1 -> Red | State2 -> Blue")
    vec1 = get_bloch_vector(rho_dm)
    vec2 = get_bloch_vector(sigma_dm)
    fig = plot_bloch_vector([vec1, vec2], title="Rho and Sigma", figsize=(3,3))
    st.pyplot(fig)