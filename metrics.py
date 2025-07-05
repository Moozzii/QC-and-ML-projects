from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.visualization import plot_bloch_vector, plot_histogram
from qiskit_aer import AerSimulator
import numpy as np
from math import sqrt
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import DensityMatrix, entropy, Pauli
from qiskit.visualization import plot_state_city

global_psi = None
global_phi = None

def check_statevectors(psi, phi):
    psi = Statevector(psi)
    phi = Statevector(phi)
    is_psi_norm = np.isclose(np.sum(np.abs(psi.data) ** 2), 1.0)
    is_phi_norm = np.isclose(np.sum(np.abs(phi.data) ** 2), 1.0)

    if is_psi_norm and is_phi_norm:
        global global_psi, global_phi
        global_psi = psi
        global_phi = phi

        return global_psi, global_phi
    else: 
        raise ValueError("Please enter statevectors whose probabilities sum to 1.")

def calculate_fidelity(psi, phi):
    fidelity = state_fidelity(psi, phi)
    return fidelity

   
def trace_distnace(psi, phi):
    fidelity = state_fidelity(psi, phi)
    calculate_trace = sqrt(1 - fidelity)
    
    return calculate_trace

def get_bloch_vector(state):
    rho = DensityMatrix(state)
    paulis = ['X', 'Y', 'Z']
    bloch = [np.real(np.trace(rho.data @ Pauli(p).to_matrix())) for p in paulis]
    return np.array(bloch)

def plot_bloch_spheres(psi, phi):
    psi_vec = get_bloch_vector(psi)
    phi_vec = get_bloch_vector(phi)

    plot_bloch_vector(psi_vec, title="Bloch Sphere for ψ")
    plot_bloch_vector(phi_vec, title="Bloch Sphere for φ")

def plot_and_simulate(statevector, qubit_index=0):
    num_qubits = int(np.log2(len(statevector)))
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.initialize(num_qubits, qubit_index)

    qc.h(qubit_index)

    qc.measure(range(num_qubits), range(num_qubits))

    simulator = AerSimulator()
    t_compile = transpile(qc, simulator)
    results = simulator.run(t_compile, shots=1024).result()
    counts = results.get_counts()

    plot_histogram(counts)


def calculate_QJSD(psi, phi):
    rho = DensityMatrix(psi)
    sigma = DensityMatrix(phi)

    average_state = (rho + sigma) / 2

    S_rho = entropy(rho)
    S_sigma = entropy(sigma)
    S_avg = entropy(average_state)

    plot_state_city(rho, title="Density Matrix of Psi", figsize=(6,6))
    plot_state_city(sigma, title="Density Matrix of Phi", figsize=(6,6))

    qjsd = S_avg - 0.5 * (S_rho + S_sigma)

    return qjsd

def is_pure(state):
    rho = DensityMatrix(state)
    purity = np.trace(rho.data @ rho.data)
    
    return np.isclose(purity, 1.0, atol=1e-6)

def compare_pure_and_mixed_states(psi, phi):
    rho = DensityMatrix(psi)
    sigma = DensityMatrix(phi)

    trace = round(trace_distnace(rho, sigma), 3)
    fidelity = round(calculate_fidelity(rho, sigma), 3)

    print(f"Fidelity: {fidelity}")
    print(f"Trace Distance: {trace}")
    print(f"F + D = {round(fidelity + trace, 3)}")

    if is_pure(psi) and is_pure(phi):
     
        lhs = trace
        rhs = round(np.sqrt(1 - fidelity), 3)
        print(f"Since both are pure: D ≈ sqrt(1 - F) = {rhs}")
        if np.isclose(lhs, rhs, atol=1e-3):
            print("✅ Metrics consistent for pure states.")
        else:
            print("❌ Something's off.")
    else:
        print("At least one state is mixed — no strict identity between D and F.")
    
    vec1 = get_bloch_vector(rho)
    vec2 = get_bloch_vector(sigma)

    plot_bloch_vector(vec1, title="Bloch Vector for ρ")
    plot_bloch_vector(vec2, title="Bloch Vector for σ")