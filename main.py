import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==================== Parameter Settings ====================
v0 = 10.0  # Free-flow speed (m/s)
a_true = -0.08  # True congestion sensitivity parameter
tau = 5.0  # Relaxation time (s)
L = 1000.0  # Yard length (m)
T = 100.0  # Time horizon (s)


# ==================== Data Generation ====================
def inflow_density(t):
    """
    Define time-varying inflow density at entrance (x=0)
    Gradually increases from low to high traffic
    """
    rho_min = 0.010  # Initial low density
    rho_max = 0.035  # Maximum density at end

    # Linear increase
    rho_inflow = rho_min + (rho_max - rho_min) * (t / T)

    # Alternative: Sigmoid smooth transition
    # rho_inflow = rho_min + (rho_max - rho_min) / (1 + np.exp(-0.1 * (t - T/2)))

    return np.clip(rho_inflow, 0.001, 0.1)


def generate_synthetic_data(nx=50, nt=50):
    """Generate synthetic trajectory data with increasing inflow"""
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    X, T_grid = np.meshgrid(x, t)

    # Initial density distribution (low initial traffic)
    rho0 = 0.012 + 0.005 * np.sin(2 * np.pi * x / L)

    # Use simplified numerical method to generate data
    rho = np.zeros((nt, nx))
    v = np.zeros((nt, nx))

    rho[0, :] = rho0
    v[0, :] = v0 * np.exp(a_true * rho0)

    dt = T / (nt - 1)
    dx = L / (nx - 1)

    for n in range(nt - 1):
        for i in range(1, nx - 1):
            # Constrain density range to avoid overflow
            rho[n, i] = np.clip(rho[n, i], 0.001, 0.1)

            # LWR equation (flow conservation)
            drho_dt = -(rho[n, i] * v[n, i] - rho[n, i - 1] * v[n, i - 1]) / dx

            # PW equation (velocity evolution)
            V_eq = v0 * np.exp(a_true * rho[n, i])
            dv_dx = (v[n, i + 1] - v[n, i - 1]) / (2 * dx)

            # Compute pressure gradient with numerical protection
            rho_avg = np.maximum(rho[n, i], 1e-6)
            dP_dx = v0 ** 2 * a_true * np.exp(a_true * rho[n, i]) * (rho[n, i + 1] - rho[n, i - 1]) / (2 * dx)

            dv_dt = -v[n, i] * dv_dx - dP_dx / rho_avg + (V_eq - v[n, i]) / tau

            # Update with smaller time step factor
            rho[n + 1, i] = np.clip(rho[n, i] + 0.5 * dt * drho_dt, 0.001, 0.1)
            v[n + 1, i] = np.clip(v[n, i] + 0.5 * dt * dv_dt, 0.1, v0)

        # Boundary conditions - KEY MODIFICATION
        # Inflow boundary (x=0): time-varying density
        current_time = t[n + 1]
        rho[n + 1, 0] = inflow_density(current_time)
        v[n + 1, 0] = v0 * np.exp(a_true * rho[n + 1, 0])  # Equilibrium velocity at inflow

        # Outflow boundary (x=L): free flow (Neumann condition)
        rho[n + 1, -1] = rho[n + 1, -2]
        v[n + 1, -1] = v[n + 1, -2]

    # Add observation noise
    rho += np.random.normal(0, 0.0005, rho.shape)
    v += np.random.normal(0, 0.05, v.shape)

    # Final clipping
    rho = np.clip(rho, 0.001, 0.1)
    v = np.clip(v, 0.1, v0)

    return X.flatten(), T_grid.flatten(), rho.flatten(), v.flatten()


# ==================== PINN Model ====================
class TrafficPINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 2]):
        super(TrafficPINN, self).__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

        # Learnable parameter a (initialized close to expected value)
        self.a = nn.Parameter(torch.tensor([-0.1], dtype=torch.float32))

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for i in range(len(self.linears) - 1):
            inputs = self.activation(self.linears[i](inputs))
        outputs = self.linears[-1](inputs)
        return outputs[:, 0:1], outputs[:, 1:2]  # rho, v


# ==================== Loss Function ====================
def compute_loss(model, x_data, t_data, rho_data, v_data, x_colloc, t_colloc):
    # Data loss
    x_d = torch.tensor(x_data, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    t_d = torch.tensor(t_data, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    rho_pred, v_pred = model(x_d, t_d)

    rho_target = torch.tensor(rho_data, dtype=torch.float32).reshape(-1, 1)
    v_target = torch.tensor(v_data, dtype=torch.float32).reshape(-1, 1)

    data_loss = torch.mean((rho_pred - rho_target) ** 2) + torch.mean((v_pred - v_target) ** 2)

    # Physics loss
    x_c = torch.tensor(x_colloc, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    t_c = torch.tensor(t_colloc, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    rho_c, v_c = model(x_c, t_c)

    # Compute derivatives
    rho_t = torch.autograd.grad(rho_c, t_c, torch.ones_like(rho_c), create_graph=True)[0]
    rho_x = torch.autograd.grad(rho_c, x_c, torch.ones_like(rho_c), create_graph=True)[0]
    v_t = torch.autograd.grad(v_c, t_c, torch.ones_like(v_c), create_graph=True)[0]
    v_x = torch.autograd.grad(v_c, x_c, torch.ones_like(v_c), create_graph=True)[0]

    # LWR equation residual
    flux = rho_c * v_c
    flux_x = torch.autograd.grad(flux, x_c, torch.ones_like(flux), create_graph=True)[0]
    lwr_residual = rho_t + flux_x

    # PW equation residual
    V_eq = v0 * torch.exp(model.a * rho_c)
    P = v0 ** 2 * torch.exp(model.a * rho_c)
    P_x = torch.autograd.grad(P, x_c, torch.ones_like(P), create_graph=True)[0]
    pw_residual = v_t + v_c * v_x + P_x / (rho_c + 1e-6) - (V_eq - v_c) / tau

    physics_loss = torch.mean(lwr_residual ** 2) + torch.mean(pw_residual ** 2)

    return data_loss + 1.0 * physics_loss, data_loss, physics_loss


# ==================== Training ====================
def train_pinn(model, x_data, t_data, rho_data, v_data, epochs=5000):
    # Use different learning rates for network weights and parameter a
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if n != 'a'], 'lr': 0.001},
        {'params': model.a, 'lr': 0.0001}  # Smaller learning rate for a
    ])

    # Generate collocation points - FIXED: create arrays separately
    n_colloc = 1000
    x_colloc = np.random.rand(n_colloc) * L
    t_colloc = np.random.rand(n_colloc) * T

    # Verify dimensions
    print(f"Collocation points: x_colloc shape = {x_colloc.shape}, t_colloc shape = {t_colloc.shape}")
    print(f"Data points: x_data shape = {x_data.shape}, t_data shape = {t_data.shape}")

    loss_history = []
    a_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, data_loss, physics_loss = compute_loss(
            model, x_data, t_data, rho_data, v_data, x_colloc, t_colloc
        )
        loss.backward()
        optimizer.step()

        # Constrain parameter a within reasonable range (wider range)
        with torch.no_grad():
            model.a.data = torch.clamp(model.a.data, -0.20, -0.01)

        loss_history.append(loss.item())
        a_history.append(model.a.item())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, "
                  f"Data Loss: {data_loss.item():.6f}, Physics Loss: {physics_loss.item():.6f}, "
                  f"a: {model.a.item():.6f}")

    return loss_history, a_history


# ==================== Visualization ====================
def visualize_results(model, X, T_grid, rho_true, v_true, loss_history, a_history):
    fig = plt.figure(figsize=(20, 14))

    # Make predictions
    with torch.no_grad():
        x_test = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
        t_test = torch.tensor(T_grid, dtype=torch.float32).reshape(-1, 1)
        rho_pred, v_pred = model(x_test, t_test)
        rho_pred = rho_pred.numpy().flatten()
        v_pred = v_pred.numpy().flatten()

    # 1. True vs Predicted Density Field
    ax1 = fig.add_subplot(3, 4, 1)
    sc1 = ax1.scatter(X, T_grid, c=rho_true, cmap='jet', s=1)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Time t (s)')
    ax1.set_title('True Density Field')
    plt.colorbar(sc1, ax=ax1, label='Density ρ')

    ax2 = fig.add_subplot(3, 4, 2)
    sc2 = ax2.scatter(X, T_grid, c=rho_pred, cmap='jet', s=1)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Time t (s)')
    ax2.set_title('Predicted Density Field')
    plt.colorbar(sc2, ax=ax2, label='Density ρ')

    # 2. True vs Predicted Velocity Field
    ax3 = fig.add_subplot(3, 4, 5)
    sc3 = ax3.scatter(X, T_grid, c=v_true, cmap='coolwarm', s=1)
    ax3.set_xlabel('Position x (m)')
    ax3.set_ylabel('Time t (s)')
    ax3.set_title('True Velocity Field')
    plt.colorbar(sc3, ax=ax3, label='Velocity v (m/s)')

    ax4 = fig.add_subplot(3, 4, 6)
    sc4 = ax4.scatter(X, T_grid, c=v_pred, cmap='coolwarm', s=1)
    ax4.set_xlabel('Position x (m)')
    ax4.set_ylabel('Time t (s)')
    ax4.set_title('Predicted Velocity Field')
    plt.colorbar(sc4, ax=ax4, label='Velocity v (m/s)')

    # 3. Congestion Level Analysis
    ax5 = fig.add_subplot(3, 4, 3)
    congestion = rho_pred / 0.1  # Normalized congestion level (assuming max density 0.1)
    congestion = np.clip(congestion * 100, 0, 100)
    sc5 = ax5.scatter(X, T_grid, c=congestion, cmap='RdYlGn_r', s=1, vmin=0, vmax=100)
    ax5.set_xlabel('Position x (m)')
    ax5.set_ylabel('Time t (s)')
    ax5.set_title('Congestion Level (%)')
    plt.colorbar(sc5, ax=ax5, label='Congestion')

    # 4. Speed-Density Relationship
    ax6 = fig.add_subplot(3, 4, 7)
    ax6.scatter(rho_true, v_true, alpha=0.3, s=1, label='True Data')
    rho_range = np.linspace(0, np.max(rho_true), 100)
    v_theoretical_true = v0 * np.exp(a_true * rho_range)
    v_theoretical_pred = v0 * np.exp(model.a.item() * rho_range)
    ax6.plot(rho_range, v_theoretical_true, 'r-', linewidth=2, label=f'True (a={a_true})')
    ax6.plot(rho_range, v_theoretical_pred, 'b--', linewidth=2, label=f'Estimated (a={model.a.item():.4f})')
    ax6.set_xlabel('Density ρ')
    ax6.set_ylabel('Velocity v (m/s)')
    ax6.set_title('Speed-Density Relationship')
    ax6.legend()
    ax6.grid(True)

    # 5. Training Loss Curve
    ax7 = fig.add_subplot(3, 4, 9)
    ax7.plot(loss_history)
    ax7.set_xlabel('Training Epoch')
    ax7.set_ylabel('Loss')
    ax7.set_title('Training Loss Curve')
    ax7.set_yscale('log')
    ax7.grid(True)

    # 6. Parameter a Convergence
    ax8 = fig.add_subplot(3, 4, 10)
    ax8.plot(a_history, label='Estimated a')
    ax8.axhline(y=a_true, color='r', linestyle='--', linewidth=2, label='True a')
    ax8.set_xlabel('Training Epoch')
    ax8.set_ylabel('Parameter a')
    ax8.set_title('Parameter a Convergence')
    ax8.legend()
    ax8.grid(True)

    # 7. Average Congestion Over Time
    ax9 = fig.add_subplot(3, 4, 11)
    nx = 50
    nt = 50
    rho_grid = rho_pred.reshape(nt, nx)
    avg_congestion = np.mean(rho_grid, axis=1) / 0.1 * 100
    t_axis = np.linspace(0, T, nt)
    ax9.plot(t_axis, avg_congestion, linewidth=2)
    ax9.fill_between(t_axis, 0, avg_congestion, alpha=0.3)
    ax9.set_xlabel('Time t (s)')
    ax9.set_ylabel('Average Congestion (%)')
    ax9.set_title('Average Yard Congestion Over Time')
    ax9.grid(True)

    # 8. NEW: Inflow Density Over Time
    ax10 = fig.add_subplot(3, 4, 4)
    t_inflow = np.linspace(0, T, 100)
    inflow_rho = [inflow_density(t) for t in t_inflow]
    ax10.plot(t_inflow, inflow_rho, linewidth=2, color='blue')
    ax10.fill_between(t_inflow, 0, inflow_rho, alpha=0.3)
    ax10.set_xlabel('Time t (s)')
    ax10.set_ylabel('Inflow Density ρ')
    ax10.set_title('Entrance Inflow Density (Increasing)')
    ax10.grid(True)

    # 9. NEW: Entrance vs Exit Density Comparison
    ax11 = fig.add_subplot(3, 4, 8)
    rho_grid_true = rho_true.reshape(nt, nx)
    t_boundary = np.linspace(0, T, nt)  # Fixed: match the time steps
    entrance_density = rho_grid_true[:, 0]
    exit_density = rho_grid_true[:, -1]
    ax11.plot(t_boundary, entrance_density, linewidth=2, label='Entrance (x=0)', color='red')
    ax11.plot(t_boundary, exit_density, linewidth=2, label='Exit (x=L)', color='green')
    ax11.set_xlabel('Time t (s)')
    ax11.set_ylabel('Density ρ')
    ax11.set_title('Entrance vs Exit Density')
    ax11.legend()
    ax11.grid(True)

    # 10. NEW: Flow Rate Over Time
    ax12 = fig.add_subplot(3, 4, 12)
    flow_rate = np.mean(rho_grid_true * v_true.reshape(nt, nx), axis=1)  # Fixed: use true data for consistency
    ax12.plot(t_boundary, flow_rate, linewidth=2, color='purple')
    ax12.fill_between(t_boundary, 0, flow_rate, alpha=0.3)
    ax12.set_xlabel('Time t (s)')
    ax12.set_ylabel('Average Flow Rate (veh/s)')
    ax12.set_title('Average Traffic Flow Rate')
    ax12.grid(True)

    plt.tight_layout()
    plt.savefig('traffic_pinn_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== Main Program ====================
if __name__ == "__main__":
    print("=" * 60)
    print("PINN-based Traffic Flow Parameter Estimation System")
    print("=" * 60)
    print(f"True parameter: a_true = {a_true}")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic data...")
    X, T_grid, rho_data, v_data = generate_synthetic_data(nx=50, nt=50)
    print(f"Number of data points: {len(X)}")
    print(f"X shape: {X.shape}, T shape: {T_grid.shape}")
    print(f"rho_data shape: {rho_data.shape}, v_data shape: {v_data.shape}")

    # Create model
    print("\nCreating PINN model...")
    model = TrafficPINN()
    print(f"Initial parameter a: {model.a.item():.6f}")

    # Train
    print("\nStarting training...")
    loss_history, a_history = train_pinn(model, X, T_grid, rho_data, v_data, epochs=5000)

    # Output results
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"True parameter a: {a_true:.6f}")
    print(f"Estimated parameter a: {model.a.item():.6f}")
    print(f"Estimation error: {abs(model.a.item() - a_true):.6f}")
    print(f"Relative error: {abs(model.a.item() - a_true) / abs(a_true) * 100:.2f}%")
    print("=" * 60)

    # Visualize
    print("\nGenerating visualization results...")
    visualize_results(model, X, T_grid, rho_data, v_data, loss_history, a_history)
    print("\nResults saved to 'traffic_pinn_results.png'")