import warnings

import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf
warnings.filterwarnings("ignore", category=UserWarning)
# --- ADD THIS TO THE TOP OF YOUR SCRIPT ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled Dynamic Memory Growth")
    except RuntimeError as e:
        print(e)
import keras
from sklearn.metrics import roc_curve, auc

# --- HPCC CLUSTER FIXES ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

PLOT_DIR = "plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# ==============================================================================
# --- 1. EXACT PHYSICS GENERATOR (Needed for 1D Mass Plots on the fly) ---
# ==============================================================================
class Particle:
    def __init__(self, mass, energy, dir):
        self.mass = np.reshape(mass, (-1, 1))
        self.energy = np.reshape(energy, (-1, 1))
        self.dir = np.reshape(dir, (-1, 3))
        
        self.dir /= np.linalg.norm(self.dir, axis=-1, keepdims=True)
        self.momentum = (np.maximum(0, self.energy**2 - self.mass**2))**.5
        self.p_3vec = self.momentum * self.dir
        self.energy = np.broadcast_to(self.energy, (self.p_3vec.shape[0], 1))
        self.p_4vec = np.concatenate([self.energy, self.p_3vec], axis=-1)
        
    def boost(self, v, dir, gamma=None):
        def dot(a, b): return np.sum(a*b, axis=-1, keepdims=True)
        if gamma is None: gamma = (1-v**2)**-.5
        v_3vec = v*dir
        energy_new = gamma * (self.energy + dot(self.p_3vec, v_3vec))
        p_3vec_new = self.p_3vec + (gamma-1) * dot(self.p_3vec, dir) * dir + gamma*self.energy*v_3vec
        dir_new = p_3vec_new / np.linalg.norm(p_3vec_new, axis=-1, keepdims=True)
        return Particle(self.mass, energy_new, dir_new)
    
    def decay(self, m1, m2, rng=None, m1_theta_com=None):
        m1 = np.reshape(m1, (-1, 1))
        m2 = np.reshape(m2, (-1, 1))
        N = max(len(self.mass), len(self.energy), len(self.dir), len(m1), len(m2))
        
        if m1_theta_com is not None:
            m1_theta_com = np.reshape(m1_theta_com, (-1, 1))
            N = max(N, len(m1_theta_com))
        else:
            m1_theta_com = np.arccos(rng.uniform(-1, 1, size=(N, 1)))
        
        safe_mass = np.maximum(self.mass, 1e-9)
        e1_com = (safe_mass**2 + m1**2 - m2**2)/(2*safe_mass)
        e2_com = (safe_mass**2 + m2**2 - m1**2)/(2*safe_mass)
        
        unnorm_random_dir = rng.standard_normal(size=(N, 3))
        def dot(a, b): return np.sum(a*b, axis=-1, keepdims=True)
        perp_dir = unnorm_random_dir - dot(unnorm_random_dir, self.dir) * self.dir
        perp_dir /= np.linalg.norm(perp_dir, axis=-1, keepdims=True)
        
        dir1_com = self.dir*np.cos(m1_theta_com) + perp_dir*np.sin(m1_theta_com)
        dir2_com = -dir1_com
        
        child1_com = Particle(m1, e1_com, dir1_com)
        child2_com = Particle(m2, e2_com, dir2_com)
        
        boost_v = (np.maximum(0, self.energy**2 - self.mass**2))**.5 / self.energy
        boost_gamma = self.energy/self.mass
        child1 = child1_com.boost(boost_v, self.dir, boost_gamma)
        child2 = child2_com.boost(boost_v, self.dir, boost_gamma)
        return child1, child2

MAX_PARTICLES = 240
M0_range = [5, 100]
rng = np.random.Generator(np.random.PCG64(42))

def generate_pair_production_dataset(N, rng, fixed_mass=None):
    params = np.full((N, 1), fixed_mass) if fixed_mass else rng.uniform(M0_range[0], M0_range[1], (N, 1))
    raw_mult = np.floor(rng.exponential(scale=3.0, size=(N,))).astype(int)
    n_A_arr = np.clip(raw_mult + 2, 2, 12)
    M_X = rng.uniform(12.5 * params, 15.0 * params, size=(N, 1))
    high_energy_bound = np.maximum(500.0, M_X * 1.5) 
    E_X = rng.uniform(M_X, high_energy_bound, size=(N, 1))
    dir_X = rng.standard_normal((N, 3))
    dir_X /= np.linalg.norm(dir_X, axis=-1, keepdims=True)
    
    events = np.zeros((N, MAX_PARTICLES, 4), dtype=np.float32)
    m_final = np.zeros_like(params)
    
    for n_A in range(2, 13):
        mask = (n_A_arr == n_A)
        if not np.any(mask): continue
        N_m = np.sum(mask)
        p_m = params[mask]
        curr_X = Particle(mass=M_X[mask], energy=E_X[mask], dir=dir_X[mask])
        
        for i in range(n_A - 1):
            if i == n_A - 2:
                A1, A2 = curr_X.decay(p_m, p_m, rng)
                A_list = [(A1, i), (A2, i+1)]
            else:
                rem_A = n_A - 1 - i
                low_bound = rem_A * p_m + 0.1
                high_bound = np.maximum(low_bound + 0.1, curr_X.mass - p_m - 0.1)
                m_next = rng.uniform(low_bound, high_bound, size=(N_m, 1))
                A1, curr_X = curr_X.decay(p_m, m_next, rng)
                A_list = [(A1, i)]
                
            for A_part, A_idx in A_list:
                raw_d_mult = np.floor(rng.exponential(scale=4.0, size=(N_m,))).astype(int)
                n_d_arr = np.clip(raw_d_mult + 2, 2, 20)
                
                for n_d in range(2, 21):
                    d_mask = (n_d_arr == n_d)
                    if not np.any(d_mask): continue
                    N_d_m = np.sum(d_mask)
                    curr_A = Particle(mass=A_part.mass[d_mask], energy=A_part.energy[d_mask], dir=A_part.dir[d_mask])
                    abs_mask = np.zeros(N, dtype=bool)
                    abs_indices = np.where(mask)[0][d_mask]
                    abs_mask[abs_indices] = True
                    
                    for j in range(n_d - 1):
                        if j == n_d - 2:
                            d1, d2 = curr_A.decay(m_final[abs_mask], m_final[abs_mask], rng)
                            events[abs_mask, A_idx * 20 + j, :]   = d1.p_4vec
                            events[abs_mask, A_idx * 20 + j + 1, :] = d2.p_4vec
                        else:
                            rem_d = n_d - 1 - j
                            low_bound = rem_d * 0.1
                            high_bound = np.maximum(low_bound + 0.1, curr_A.mass - 0.1)
                            m_next = rng.uniform(low_bound, high_bound, size=(N_d_m, 1))
                            d1, curr_A = curr_A.decay(m_final[abs_mask], m_next, rng)
                            events[abs_mask, A_idx * 20 + j, :] = d1.p_4vec

    keys = rng.random((N, MAX_PARTICLES))
    keys[np.sum(np.abs(events), axis=-1) == 0] = 10.0 
    final_events = events[np.arange(N)[:, None], keys.argsort(axis=1)]
    return params, final_events, M_X

# ==============================================================================
# --- 2. LOAD DATA & MODELS ---
# ==============================================================================
print("\nLoading cached dataset...")
DATA_FILE = "/lustre/research/hep/akshriva/SVJ_RandD/Darkpion_EVN/generated_datamultiplicity_dataset/svj_dataset_max240.npz"
cached_data = np.load(DATA_FILE)
test_p = cached_data['test_p']
test_vis = cached_data['test_vis']

print("\nLoading trained models from disk...")
std_m = keras.saving.load_model("/lustre/research/hep/akshriva/SVJ_RandD/Darkpion_EVN/saved_models/MEGAstd_full_classifier.keras")
std_e = keras.saving.load_model("/lustre/research/hep/akshriva/SVJ_RandD/Darkpion_EVN/saved_models/MEGAstd_encoder.keras")
print("[SUCCESS] Data and Models loaded! Moving straight to evaluation.")

# ==============================================================================
# --- 3. PERFORMANCE DIAGNOSTICS & BASE INFERENCE ---
# ==============================================================================
print("\nGenerating Analysis Results...")

# Create a temporary model that ONLY outputs the 1D score (ignores the massive attention weights)
score_extractor = keras.Model(inputs=std_e.input, outputs=std_e.output[0])

# Now we can safely predict all 50,000 events (batch size can even go back up!)
std_scores = score_extractor.predict(test_vis, batch_size=512, verbose=0).squeeze()

flip_std = scipy.stats.kendalltau(std_scores, test_p.flatten()).correlation < 0
if flip_std: std_scores *= -1

std_tau = scipy.stats.kendalltau(std_scores, test_p.flatten()).correlation

print("\n" + "="*50)
print("   INFERENCE VERIFICATION & INPUT MATRIX (FIRST 4 EVENTS)")
print("="*50)
for i in range(4):
    print(f"\n--- EVENT {i+1} ---")
    print(f"Target Mass (m_A) : {test_p[i,0]:.2f} GeV")
    
    raw_input = test_vis[i]
    log_scaled = np.sign(raw_input) * np.log1p(np.abs(raw_input) / 10.0)
    
    print("Raw Input (E, px, py, pz):")
    print(np.round(raw_input, 2))
    print("\nLog-Scaled Input Matrix (Fed to Transformer):")
    print(np.round(log_scaled, 3))
    print(f"\nStandard Extracted V  : {std_scores[i]:.4f}")
# ==============================================================================
# --- 4. PLOTTING: DIAGNOSTICS & FEYNMAN DIAGRAM ---
# ==============================================================================
print("Generating Particle Input Matrix Plot...")
log_scaled_event0 = np.sign(test_vis[0]) * np.log1p(np.abs(test_vis[0]) / 10.0)

fig_in, ax_in = plt.subplots(figsize=(6, 8))
im = ax_in.imshow(log_scaled_event0, cmap='coolwarm', aspect='auto')
ax_in.set_title("Transformer Input Tensor (Event 1)", fontsize=14)
ax_in.set_ylabel("Particle Index (1-8)", fontsize=12)
ax_in.set_xticks([0, 1, 2, 3])
ax_in.set_xticklabels(["E", "$p_x$", "$p_y$", "$p_z$"], fontsize=12)
for i in range(8):
    for j in range(4):
        ax_in.text(j, i, f"{log_scaled_event0[i, j]:.2f}", ha="center", va="center", color="white" if abs(log_scaled_event0[i,j]) > 1.5 else "black")
fig_in.colorbar(im, ax=ax_in, label="Log-Scaled Magnitude")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/input_tensor_visualization.pdf")

print("Generating Isolated Variable Multiplicity Feynman Diagram...")
fig_feyn, ax_feyn = plt.subplots(figsize=(12, 7))
ax_feyn.axis('off')
ax_feyn.set_title("Topology: $X \\to nA \\to \\sum m_i$ Particles (Variable Sequential Cascades)", fontsize=18, y=0.95)

def draw_line(ax, p1, p2, text=None, text_offset=(0, 0.04)):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=2)
    if text:
        ax.text((p1[0]+p2[0])/2 + text_offset[0], (p1[1]+p2[1])/2 + text_offset[1], 
                text, fontsize=14, ha='center', va='center')

def draw_A_sequential(ax, sx, sy, A_idx, num_daughters, width=2.0):
    step_x = width / (num_daughters - 0.5)
    curr_x, curr_y = sx, sy
    for d in range(num_daughters - 1):
        next_x = curr_x + step_x
        if d == num_daughters - 2:
            draw_line(ax, (curr_x, curr_y), (next_x, curr_y + 0.15), f"$p_{{{A_idx},{d+1}}}$")
            draw_line(ax, (curr_x, curr_y), (next_x, curr_y - 0.15), f"$p_{{{A_idx},{d+2}}}$", text_offset=(0, -0.05))
        else:
            next_y_remnant = curr_y - 0.08
            draw_line(ax, (curr_x, curr_y), (next_x, curr_y + 0.15), f"$p_{{{A_idx},{d+1}}}$")
            draw_line(ax, (curr_x, curr_y), (next_x, next_y_remnant), "") 
            curr_x, curr_y = next_x, next_y_remnant

draw_line(ax_feyn, (0, 0.5), (1.5, 0.5), "$X$")
draw_line(ax_feyn, (1.5, 0.5), (3.0, 0.8), "$A_1$")          
draw_line(ax_feyn, (1.5, 0.5), (3.0, 0.2), "$X_{remnant}$")  
draw_line(ax_feyn, (3.0, 0.2), (4.5, 0.4), "$A_2$")        
draw_line(ax_feyn, (3.0, 0.2), (4.5, -0.2), "$A_3$")       

draw_A_sequential(ax_feyn, 3.0, 0.8, A_idx=1, num_daughters=3, width=2.5)  
draw_A_sequential(ax_feyn, 4.5, 0.4, A_idx=2, num_daughters=5, width=3.5)  
draw_A_sequential(ax_feyn, 4.5, -0.2, A_idx=3, num_daughters=2, width=1.5) 

ax_feyn.set_xlim(0, 8.5)
ax_feyn.set_ylim(-0.6, 1.2)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/isolated_feynman_diagram.pdf")

print("Generating Attention Maps...")
evt_idx = np.random.randint(0, len(test_vis))

# Grab just ONE event and add the batch dimension back (shape: 1, 240, 4)
single_event = test_vis[evt_idx:evt_idx+1] 

# Predict using the full encoder to get the weights for just this one event
_, single_weights = std_e.predict(single_event, verbose=0)
single_weights = single_weights[0] # remove batch dimension

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
fig.suptitle(f"Attention Weight Allocation (Event {evt_idx})", fontsize=16)

# Plot the mean across all 8 attention heads
im0 = ax.imshow(np.mean(single_weights, axis=0), cmap='hot', interpolation='nearest', vmin=0, vmax=0.3)
ax.set_title("Standard Attention Matrix"); ax.set_xlabel("Key Particle"); ax.set_ylabel("Query Particle")
fig.colorbar(im0, ax=ax)

ax.set_xticks(np.arange(0, MAX_PARTICLES, 8))
ax.set_yticks(np.arange(0, MAX_PARTICLES, 8))
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/pairprod_attention.pdf")
# ==============================================================================
# --- 6. PLOTTING: ADVANCED EVALUATION (ROC & PROFILE) ---
# ==============================================================================
print("Generating ROC and Profile Plots...")
roc_p_true = test_p
roc_p_fake = rng.uniform(M0_range[0], M0_range[1], test_p.shape)

roc_p = np.concatenate([roc_p_true, roc_p_fake])
roc_vis = np.concatenate([test_vis, test_vis])
roc_y = np.concatenate([np.ones(len(test_p)), np.zeros(len(test_p))])

pred_std = std_m.predict([roc_p, roc_vis], batch_size=64, verbose=0).squeeze()
fpr_std, tpr_std, _ = roc_curve(roc_y, pred_std)
auc_std = auc(fpr_std, tpr_std)

fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
ax_roc.plot(fpr_std, tpr_std, color='blue', lw=2, label=f'Standard Attention (AUC = {auc_std:.4f})')
ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (Fake Mass Assignment)', fontsize=14)
ax_roc.set_ylabel('True Positive Rate (True Mass Assignment)', fontsize=14)
ax_roc.set_title('Classifier ROC Curve (Parameterized Background vs Signal)', fontsize=16)
ax_roc.legend(loc="lower right", fontsize=12)
ax_roc.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/roc_curves.pdf")

print("Generating Classifier Score Distributions...")
scores_class1 = pred_std[roc_y == 1]
scores_class0 = pred_std[roc_y == 0]

fig_score, ax_score = plt.subplots(figsize=(8, 6))
ax_score.hist(scores_class1, bins=50, alpha=0.5, color='blue', label='True Pairings (Class 1)', density=True)
ax_score.hist(scores_class0, bins=50, alpha=0.5, color='red', label='Fake Pairings (Class 0)', density=True)
ax_score.hist(scores_class1, bins=50, color='blue', histtype='step', lw=1.5, density=True)
ax_score.hist(scores_class0, bins=50, color='red', histtype='step', lw=1.5, density=True)

ax_score.set_title("Classifier Output Scores (Standard Attention)", fontsize=16)
ax_score.set_xlabel("Predicted Probability (Sigmoid Output)", fontsize=14)
ax_score.set_ylabel("Density", fontsize=14)
ax_score.legend(loc='upper center', fontsize=12)
ax_score.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/classifier_scores_distribution.pdf")


# ==============================================================================
# --- INVESTIGATING THE 0.8 SPIKE: SCORES BY MASS REGION ---
# ==============================================================================
print("Investigating score spike across mass regions...")

mass_points = [10, 50, 90] # Low, Mid, High
colors = ['purple', 'green', 'red']

fig_split, ax_split = plt.subplots(figsize=(9, 6))

for m, c in zip(mass_points, colors):
    # Generate pure signal at this mass
    _, v_signal, _ = generate_pair_production_dataset(5000, rng, fixed_mass=m)
    
    # Create the correct mass parameter input for the full classifier
    p_param = np.full((5000, 1), m)
    
    # Predict probabilities (using full model std_m)
    # We use std_m here because it only outputs 1 value (probability), so no OOM!
    signal_scores = std_m.predict([p_param, v_signal], batch_size=256, verbose=0).squeeze()
    
    ax_split.hist(signal_scores, bins=50, histtype='step', lw=2, color=c, 
                  label=f'Signal $m_A = {m}$ GeV', density=True)

ax_split.set_title("Classifier Confidence vs. Resonance Mass", fontsize=16)
ax_split.set_xlabel("Classifier Score (Sigmoid Output)", fontsize=14)
ax_split.set_ylabel("Density", fontsize=14)
ax_split.legend()
ax_split.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/score_spike_investigation.pdf")

print("Generating Profile Plots...")
bins = np.linspace(5, 100, 20)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
mean_std, bin_edges, _ = scipy.stats.binned_statistic(test_p.flatten(), std_scores, statistic='mean', bins=bins)
std_dev_std, _, _ = scipy.stats.binned_statistic(test_p.flatten(), std_scores, statistic='std', bins=bins)

fig_prof_std, ax_prof_std = plt.subplots(figsize=(8, 6))
ax_prof_std.errorbar(bin_centers, mean_std, yerr=std_dev_std, fmt='-o', color='blue', capsize=5, 
                 label=f'V ($\\tau$={std_tau:.3f})', alpha=0.8)
ax_prof_std.set_xlabel(r"True Resonance Mass $m_A$ (GeV)", fontsize=14)
ax_prof_std.set_ylabel(r"Mean Extracted Variable $\langle V \rangle$", fontsize=14)
ax_prof_std.set_title(" Calibration Profile", fontsize=16)
ax_prof_std.legend(loc='upper left', fontsize=12)
ax_prof_std.grid(alpha=0.3)
fig_prof_std.tight_layout()
fig_prof_std.savefig(f"{PLOT_DIR}/profile_standard.pdf")



# ==============================================================================
# --- 1D DISTRIBUTION OF RAW VARIABLE V (WITH STATS) ---
# ==============================================================================
print("Generating 1D Distributions of V with statistics...")

fig_v, ax_v = plt.subplots(figsize=(10, 7))
plot_masses = [10, 20, 30, 60, 70, 90]
colors_list = ['purple', 'darkblue', 'dodgerblue', 'gold', 'darkorange', 'crimson']

for m, c in zip(plot_masses, colors_list):
    _, v_d, _ = generate_pair_production_dataset(10000, rng, fixed_mass=m)
    raw_v_std = score_extractor.predict(v_d, batch_size=512, verbose=0).squeeze()
    if flip_std: raw_v_std *= -1
    
    # Calculate Stats
    mu_v = np.mean(raw_v_std)
    sig_v = np.std(raw_v_std)
    
    label_str = f"$m_A$={m}: $\mu_V$={mu_v:.2f}, $\sigma_V$={sig_v:.2f}"
    ax_v.hist(raw_v_std, bins=100, density=True, histtype='step', color=c, label=label_str, lw=2.0)

ax_v.set_title("Latent Variable Separation (Standard Attention)", fontsize=15)
ax_v.set_xlabel("Artificial Variable $V$", fontsize=14); ax_v.set_ylabel("Density")
ax_v.legend(frameon=False, loc='upper right', fontsize=9, ncol=1)
ax_v.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/raw_V_stats.pdf")
# ==============================================================================
# --- 2D CORRELATION HEATMAP WITH SPLINE FIT ---
# ==============================================================================
print("Generating 2D Correlation Heatmap with Spline Trendline...")

fig_heat, ax_heat = plt.subplots(figsize=(10, 8))

# 1. Plot 2D histogram (using magma for high contrast)
h = ax_heat.hist2d(std_scores, test_p.flatten(), bins=100, cmap='magma', cmin=1)

# 2. Add Colorbar
cb = fig_heat.colorbar(h[3], ax=ax_heat)
cb.set_label('Number of Events', fontsize=12)

# 3. Plot the Spline Trendline
# We calculate a fine grid of V scores to make the spline look perfectly smooth
v_fine = np.linspace(std_scores.min(), std_scores.max(), 500)
m_fine = calib_std(v_fine)

ax_heat.plot(v_fine, m_fine, color='cyan', linestyle='-', lw=3, label='Binned Spline Fit')

# 4. Formatting
ax_heat.set_title(fr"Standard Attention: Extracted $V$ vs. True $m_A$" + "\n" + fr"Kendall $\tau = {std_tau:.3f}$", fontsize=16)
ax_heat.set_xlabel(r"Artificial Variable $V$ (AI Latent Score)", fontsize=14)
ax_heat.set_ylabel(r"True Resonance Mass $m_A$ [GeV]", fontsize=14)

ax_heat.set_ylim(M0_range[0], M0_range[1])
ax_heat.legend(loc='upper left', frameon=True, facecolor='white')
ax_heat.grid(alpha=0.1)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/correlation_heatmap_SPLINE.pdf")


# ==============================================================================
# --- PROFILE PLOT (Binned Means and Std Dev) ---
# ==============================================================================
print("Generating Profile Plot...")

# 1. Define mass bins (same range as your training/testing)
bins = np.linspace(5, 100, 20)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# 2. Calculate the Mean and Standard Deviation of V in each mass bin
mean_v, bin_edges, _ = scipy.stats.binned_statistic(
    test_p.flatten(), std_scores, statistic='mean', bins=bins
)
std_v, _, _ = scipy.stats.binned_statistic(
    test_p.flatten(), std_scores, statistic='std', bins=bins
)

# 3. Plotting
fig_prof, ax_prof = plt.subplots(figsize=(8, 6))

ax_prof.errorbar(bin_centers, mean_v, yerr=std_v, fmt='-o', color='blue', 
                 capsize=5, elinewidth=1.5, markeredgewidth=1.5,
                 label=f'Standard Attention ($\\tau$={std_tau:.3f})', alpha=0.8)

# Formatting
ax_prof.set_title("AI Calibration Profile: Mean $V$ vs. True $m_A$", fontsize=16)
ax_prof.set_xlabel(r"True Resonance Mass $m_A$ (GeV)", fontsize=14)
ax_prof.set_ylabel(r"Mean Extracted Variable $\langle V \rangle$", fontsize=14)
ax_prof.legend(loc='upper left', fontsize=12)
ax_prof.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/profile_plot_V_vs_Mass.pdf")

from scipy.interpolate import UnivariateSpline

from scipy.interpolate import UnivariateSpline

# ==============================================================================
# --- 7. FAST BINNED SPLINE CALIBRATION ---
# ==============================================================================
print("Calibrating via Binned Spline (Fast)...")

# 1. Bin the data to reduce 50k points down to 100
bins = np.linspace(std_scores.min(), std_scores.max(), 100)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
binned_m, _, _ = scipy.stats.binned_statistic(std_scores, test_p.flatten(), statistic='mean', bins=bins)

# 2. Filter out empty bins (NaNs)
valid = ~np.isnan(binned_m)
# 3. Fit spline to the profile centers (s=10 is usually plenty for 100 points)
calib_std = UnivariateSpline(bin_centers[valid], binned_m[valid], s=10)

print("[SUCCESS] Spline calibrated in milliseconds.")

# ==============================================================================
# --- 8. RECONSTRUCTED MASS SPECTRA (WITH STATS) ---
# ==============================================================================
print("Generating Mass Spectra for 10, 20, 30, 60, 70, 90 GeV...")
fig_reco, ax_reco = plt.subplots(figsize=(12, 8))

target_masses = [10, 20, 30, 60, 70, 90]
colors = ['purple', 'darkblue', 'dodgerblue', 'gold', 'darkorange', 'crimson']

for m, c in zip(target_masses, colors):
    # Generate 10k events per mass point
    _, v_d, _ = generate_pair_production_dataset(10000, rng, fixed_mass=m)
    
    # Predict raw V using the extractor
    raw_v = score_extractor.predict(v_d, batch_size=512, verbose=0).squeeze()
    if flip_std: raw_v *= -1
    
    # Map to GeV using the Spline
    reco_m = calib_std(raw_v)
    
    # Calculate Stats
    mu = np.mean(reco_m)
    sigma = np.std(reco_m)
    resolution = (sigma / m) * 100
    
    label_text = f"True {m} GeV: $\mu$={mu:.1f}, $\sigma$={sigma:.1f} ({resolution:.1f}%)"
    
    # Plot
    ax_reco.hist(reco_m, bins=100, range=(0, 130), density=True, histtype='step', 
                 color=c, label=label_text, lw=2.0)
    ax_reco.axvline(m, color=c, linestyle=':', alpha=0.5)

ax_reco.set_title("Spline-Calibrated Mass Reconstruction with Resolution Stats", fontsize=16)
ax_reco.set_xlabel("Reconstructed Mass $m_{reco}$ (GeV)", fontsize=14)
ax_reco.set_ylabel("Density", fontsize=14)
ax_reco.legend(frameon=False, loc='upper right', fontsize=10)
ax_reco.grid(alpha=0.2)
ax_reco.set_xlim(0, 130)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/reconstructed_mass_spectra_FINAL.pdf")


# ==============================================================================
# --- 9. RESOLUTION & BIAS SUMMARY TRENDS ---
# ==============================================================================
print("Generating Performance Summary (Resolution & Bias)...")

# Define a fine grid of masses for the summary
scan_masses = np.arange(10, 101, 5)
resolutions = []
biases = []

for m in scan_masses:
    # Generate test sample for this mass
    _, v_d, _ = generate_pair_production_dataset(5000, rng, fixed_mass=m)
    
    # Get scores and map to GeV
    raw_v = score_extractor.predict(v_d, batch_size=512, verbose=0).squeeze()
    if flip_std: raw_v *= -1
    reco_m = calib_std(raw_v)
    
    # Calculate % resolution: sigma/m
    res = (np.std(reco_m) / m) * 100
    resolutions.append(res)
    
    # Calculate % bias: (mean - true)/true
    bias = ((np.mean(reco_m) - m) / m) * 100
    biases.append(bias)

# --- PLOTTING ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left Y-axis: Resolution
color1 = 'tab:blue'
ax1.set_xlabel('True Resonance Mass $m_A$ (GeV)', fontsize=14)
ax1.set_ylabel('Mass Resolution $\sigma(m)/m$ [%]', color=color1, fontsize=14)
ax1.plot(scan_masses, resolutions, 'o-', color=color1, lw=2.5, label='Resolution')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(alpha=0.3)

# Right Y-axis: Bias
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Relative Bias $\Delta m/m$ [%]', color=color2, fontsize=14)
ax2.plot(scan_masses, biases, 's--', color=color2, alpha=0.6, label='Bias')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.axhline(0, color='black', lw=1, alpha=0.5) # Zero line
ax2.set_ylim(-5, 5) # We want the bias to be near zero!

plt.title('Transformer Reconstruction Performance Summary', fontsize=16)
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}/resolution_bias_summary.pdf")
print(f"\n[FINAL DONE] All results saved to {PLOT_DIR}/")
print("\n[DONE] All plots generated successfully!")
print("\n[DONE] All plots generated and saved successfully!")
