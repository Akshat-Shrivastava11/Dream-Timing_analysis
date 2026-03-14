import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import ops
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import roc_curve, auc
# --- HPCC CLUSTER FIXES ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
tf.config.optimizer.set_jit(False)

PLOT_DIR = "plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# --- 1. EXACT PHYSICS GENERATOR ---
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
        def dot(a, b):
            return np.sum(a*b, axis=-1, keepdims=True)
        
        if gamma is None:
            gamma = (1-v**2)**-.5
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

def get_physics_matrix(p4_array):
    """Calculates pairwise invariant mass in chunks to prevent OOM Kills."""
    chunk_size = 10_000  # Process 10,000 events at a time
    masses = []
    
    for i in range(0, len(p4_array), chunk_size):
        # Force float32 to keep the memory footprint strictly bounded
        chunk = p4_array[i:i+chunk_size].astype(np.float32) 
        print(f"Processing events {i} to {min(i+chunk_size, len(p4_array))} for physics matrix...")
        p4_1 = np.expand_dims(chunk, 2)
        p4_2 = np.expand_dims(chunk, 1)
        p4_sum = p4_1 + p4_2
        
        mass = np.sqrt(np.maximum(0, p4_sum[...,0]**2 - np.sum(p4_sum[...,1:]**2, -1)))
        masses.append(np.expand_dims(mass, -1))
        
    return np.concatenate(masses, axis=0)

# --- 2. CONFIGURATION & SYMMETRIC 4-BODY CASCADE ---
# --- 2. CONFIGURATION & DUAL FALLING SPECTRUM CASCADE (2-12 A's, 2-20 Daughters) ---
N_train, N_test = 300_000, 50_000
MAX_PARTICLES = 240  # Up to 12 A's * up to 20 daughters each = 240 max particles
M0_range = [5, 200]  # mA strictly 5 to 100 GeV

def generate_pair_production_dataset(N, rng, fixed_mass=None):
    params = np.full((N, 1), fixed_mass) if fixed_mass else rng.uniform(M0_range[0], M0_range[1], (N, 1))
    
    # 1. FALLING SPECTRUM for A particles (Capped at 12)
    # Using scale=3.0 so the exponential drop-off fits nicely within the 2-12 range
    raw_mult = np.floor(rng.exponential(scale=3.0, size=(N,))).astype(int)
    n_A_arr = np.clip(raw_mult + 2, 2, 12)
    
    # 2. X scaled back to safely decay into up to 12 A's
    M_X = rng.uniform(12.5 * params, 15.0 * params, size=(N, 1))
    high_energy_bound = np.maximum(500.0, M_X * 1.5) 
    E_X = rng.uniform(M_X, high_energy_bound, size=(N, 1))
    
    dir_X = rng.standard_normal((N, 3))
    dir_X /= np.linalg.norm(dir_X, axis=-1, keepdims=True)
    
    events = np.zeros((N, MAX_PARTICLES, 4), dtype=np.float32)
    m_final = np.zeros_like(params)
    
    # Iterate through the upper bound of 12 A particles
    for n_A in range(2, 13):
        mask = (n_A_arr == n_A)
        if not np.any(mask): continue
        
        N_m = np.sum(mask)
        p_m = params[mask]
        
        curr_X = Particle(mass=M_X[mask], energy=E_X[mask], dir=dir_X[mask])
        
        # 3. Sequentially cascade X into n_A particles
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
                
            # 4. FALLING SPECTRUM for Daughters (Constituents)
            for A_part, A_idx in A_list:
                # Exponential drop-off for constituents, capped at 20
                raw_d_mult = np.floor(rng.exponential(scale=4.0, size=(N_m,))).astype(int)
                n_d_arr = np.clip(raw_d_mult + 2, 2, 20)
                
                for n_d in range(2, 21):
                    d_mask = (n_d_arr == n_d)
                    if not np.any(d_mask): continue
                    
                    N_d_m = np.sum(d_mask)
                    
                    curr_A = Particle(
                        mass=A_part.mass[d_mask], 
                        energy=A_part.energy[d_mask], 
                        dir=A_part.dir[d_mask]
                    )
                    
                    abs_mask = np.zeros(N, dtype=bool)
                    abs_indices = np.where(mask)[0][d_mask]
                    abs_mask[abs_indices] = True
                    
                    # Sequentially decay the A particle
                    for j in range(n_d - 1):
                        # A_idx * 20 allocates a 20-slot block for each A particle
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

    # PERFECT SHUFFLE: Randomize order while pushing zero-padding to the absolute back
    keys = rng.random((N, MAX_PARTICLES))
    keys[np.sum(np.abs(events), axis=-1) == 0] = 10.0 
    final_events = events[np.arange(N)[:, None], keys.argsort(axis=1)]
    
    return params, final_events, M_X
# --- 3. DATA PREP, CHUNKING & CACHING ---
DATA_DIR = "generated_datamultiplicity_dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DATA_FILE = f"{DATA_DIR}/svj_dataset_max240.npz"
rng = np.random.Generator(np.random.PCG64(42))

def generate_in_chunks(N, rng_engine, chunk_size=25_000):
    """Generates dataset in smaller memory-safe chunks."""
    all_p = np.zeros((N, 1), dtype=np.float32)
    all_vis = np.zeros((N, MAX_PARTICLES, 4), dtype=np.float32)
    all_MX = np.zeros((N, 1), dtype=np.float32)
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        print(f"  -> Processing events {i} to {end}...")
        p_c, vis_c, MX_c = generate_pair_production_dataset(end - i, rng_engine)
        
        all_p[i:end] = p_c
        all_vis[i:end] = vis_c
        all_MX[i:end] = MX_c
        
        # Force a memory dump to keep RAM flat
        del p_c, vis_c, MX_c
        import gc
        gc.collect()
        
    return all_p, all_vis, all_MX

if os.path.exists(DATA_FILE):
    print(f"Loading cached dataset from {DATA_FILE}...")
    cached_data = np.load(DATA_FILE)
    
    p_train = cached_data['p_train']
    vis_train = cached_data['vis_train']
    y_train = cached_data['y_train']
    
    test_p = cached_data['test_p']
    test_vis = cached_data['test_vis']
    test_MX = cached_data['test_MX']
    
    # We save these specifically to reconstruct the Prior Plots
    c1_p = cached_data['c1_p']
    c1_MX = cached_data['c1_MX']
    
else:
    print("Generating new chunked dataset to protect RAM...")
    print("1/3: Generating Class 1 (True Signal)...")
    c1_p, c1_vis, c1_MX = generate_in_chunks(N_train, rng)
    
    print("2/3: Generating Class 0 (Fake Background)...")
    c0_p = rng.uniform(M0_range[0], M0_range[1], (N_train, 1)).astype(np.float32)
    c0_vis = c1_vis # Direct physical copy
    
    # Combine and shuffle
    print("Merging and Shuffling Training Sets...")
    p_train = np.concatenate([c0_p, c1_p])
    vis_train = np.concatenate([c0_vis, c1_vis])
    y_train = np.concatenate([np.zeros(N_train, dtype=np.float32), np.ones(N_train, dtype=np.float32)])
    
    perm = rng.permutation(2*N_train)
    p_train, vis_train, y_train = p_train[perm], vis_train[perm], y_train[perm]
    
    print("3/3: Generating Test Set...")
    test_p, test_vis, test_MX = generate_in_chunks(N_test, rng)
    
    print(f"Saving dataset to disk at {DATA_FILE}...")
    np.savez_compressed(
        DATA_FILE, 
        p_train=p_train, vis_train=vis_train, y_train=y_train,
        test_p=test_p, test_vis=test_vis, test_MX=test_MX,
        c1_p=c1_p, c1_MX=c1_MX
    )
    print("Save complete!")

# KINEMATIC & MASS PRIOR PLOTS
print("Generating Prior Kinematic & Multiplicity Plots...")

# Figure A: Mass Spectra
fig_mass, ax_mass = plt.subplots(1, 2, figsize=(14, 5))
fig_mass.suptitle("Generated Mass Spectra", fontsize=18)
ax_mass[0].hist(c1_p, bins=50, color='royalblue', alpha=0.7, histtype='stepfilled')
ax_mass[0].set_title("True Resonance Mass $m_A$ (5-100 GeV)")
ax_mass[0].set_xlabel("Mass (GeV)")
ax_mass[0].set_ylabel("Counts")

ax_mass[1].hist(c1_MX, bins=50, color='forestgreen', alpha=0.7, histtype='stepfilled')
ax_mass[1].set_title("Heavy Parent Mass $M_X$ (Variable Multiplicity)")
ax_mass[1].set_xlabel("Mass (GeV)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/priors_mass_spectra.pdf")

# Figure B: Event-Level Kinematics & Multiplicities (2x2 Grid)
fig_kin, axes_kin = plt.subplots(2, 2, figsize=(18, 12))
fig_kin.suptitle("Event Kinematics & Multiplicity Distributions", fontsize=20)

axes_flat = axes_kin.flatten()

# 1. Total Multiplicity Plot (Top Left)
multiplicity = np.sum(test_vis[:, :, 0] > 0, axis=1)
axes_flat[0].hist(multiplicity, bins=np.arange(3.5, 245.5, 8), color='purple', alpha=0.7, rwidth=0.8)
axes_flat[0].set_title("Total Visible Particles per Event")
axes_flat[0].set_xlabel("Total Number of Particles")
axes_flat[0].set_ylabel("Counts")
axes_flat[0].set_xticks(np.arange(8, 248, 24))

# 2. Constituents per Dark Hadron (Top Right)
raw_d_mult_prior = np.floor(rng.exponential(scale=4.0, size=200_000)).astype(int)
n_d_prior = np.clip(raw_d_mult_prior + 2, 2, 20)

axes_flat[1].hist(n_d_prior, bins=np.arange(1.5, 21.5, 1), color='tab:red', histtype='step', lw=2.5)
axes_flat[1].set_title("Falling Spectrum: Constituents per Dark Hadron ($A$)")
axes_flat[1].set_xlabel("$n_{\\text{Dark Hadron Constituents}}$") # Updated label
axes_flat[1].set_ylabel("Counts") # Updated label
axes_flat[1].set_xticks(np.arange(2, 22, 2))

# 3. Total Visible Energy HT (Bottom Left)
ht = np.sum(test_vis[:, :, 0], axis=1)
axes_flat[2].hist(ht, bins=80, color='crimson', alpha=0.7)
axes_flat[2].set_title("Total Visible Energy ($H_T$)")
axes_flat[2].set_xlabel("Energy (GeV)")
axes_flat[2].set_ylabel("Counts")

# 4. Inclusive Overlaid px, py, pz (Bottom Right)
valid_mask = test_vis[:, :, 0] > 0
px_all = test_vis[:, :, 1][valid_mask]
py_all = test_vis[:, :, 2][valid_mask]
pz_all = test_vis[:, :, 3][valid_mask]

axes_flat[3].hist(px_all, bins=100, color='blue', alpha=0.4, histtype='stepfilled', label='$p_x$')
axes_flat[3].hist(py_all, bins=100, color='orange', alpha=0.4, histtype='stepfilled', label='$p_y$')
axes_flat[3].hist(pz_all, bins=100, color='green', alpha=0.4, histtype='stepfilled', label='$p_z$')

axes_flat[3].hist(px_all, bins=100, color='blue', histtype='step', lw=1.5)
axes_flat[3].hist(py_all, bins=100, color='orange', histtype='step', lw=1.5)
axes_flat[3].hist(pz_all, bins=100, color='green', histtype='step', lw=1.5)

axes_flat[3].set_title("Inclusive Particle Momenta (All Valid Particles)")
axes_flat[3].set_xlabel("Momentum (GeV/c)")
axes_flat[3].set_xlim(-200, 200)
axes_flat[3].legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/priors_kinematics_and_multiplicity.pdf")

# --- 4. MODEL COMPONENTS ---
@keras.saving.register_keras_serializable()
class KinematicAttention(keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=16, return_attention_scores=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads, self.key_dim, self.return_attention_scores = num_heads, key_dim, return_attention_scores
        self.d_model = num_heads * key_dim

    def build(self, input_shape):
        self.wq = keras.layers.Dense(self.d_model)
        self.wk = keras.layers.Dense(self.d_model)
        self.wv = keras.layers.Dense(self.d_model)
        self.w_physics = keras.layers.Dense(self.num_heads, kernel_initializer='zeros', bias_initializer='zeros')
        self.wo = keras.layers.Dense(self.d_model)

    def call(self, inputs, mask=None):
        x, phys = inputs; N = ops.shape(x)[1]
        q = ops.transpose(ops.reshape(self.wq(x), [-1, N, self.num_heads, self.key_dim]), [0, 2, 1, 3])
        k = ops.transpose(ops.reshape(self.wk(x), [-1, N, self.num_heads, self.key_dim]), [0, 2, 1, 3])
        v = ops.transpose(ops.reshape(self.wv(x), [-1, N, self.num_heads, self.key_dim]), [0, 2, 1, 3])
        
        scores = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) / ops.sqrt(ops.cast(self.key_dim, "float32"))
        scores += ops.transpose(self.w_physics(phys), [0, 3, 1, 2])
        
        if mask is not None:
            m = mask[0] if isinstance(mask, list) else mask
            if m is not None:
                scores = ops.where(ops.cast(ops.expand_dims(m, 1), "bool"), scores, -1e9)
            
        w = keras.layers.Softmax(axis=-1)(scores)
        out = ops.reshape(ops.transpose(ops.matmul(w, v), [0, 2, 1, 3]), [-1, N, self.d_model])
        out = self.wo(out)
        
        return (out, w) if getattr(self, 'return_attention_scores', False) else out

    def compute_output_shape(self, input_shape):
        x_s = input_shape[0]; out_s = (x_s[0], x_s[1], self.d_model)
        if self.return_attention_scores: return out_s, (x_s[0], self.num_heads, x_s[1], x_s[1])
        return out_s

def build_transformer(use_kinematic=False):
    keras.utils.set_random_seed(42)
    vis_inp = keras.Input(shape=(MAX_PARTICLES, 4), name="Visible_Particles")
    phys_inp = keras.Input(shape=(MAX_PARTICLES, MAX_PARTICLES, 1), name="Physics_Matrix") if use_kinematic else None
    
    mask_1d = ops.any(ops.not_equal(vis_inp, 0.0), axis=-1)
    attn_mask = ops.logical_and(ops.expand_dims(mask_1d, 1), ops.expand_dims(mask_1d, 2))
    
    # Ensure your initial dense layer is set to 512
    vis_scaled = ops.sign(vis_inp) * ops.log1p(ops.abs(vis_inp) / 10.0)
    x = keras.layers.Dense(512, activation='relu')(vis_scaled)
    
    if use_kinematic:
        p_scaled = ops.log1p(phys_inp / 10.0)
        # CHANGE: key_dim increased to 64 so that 8 * 64 = 512
        attn_out, weights = KinematicAttention(num_heads=8, key_dim=64, return_attention_scores=True)([x, p_scaled], mask=[attn_mask, None])
    else:
        # CHANGE: key_dim increased to 64 so that 8 * 64 = 512
        attn_layer = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        attn_out, weights = attn_layer(query=x, value=x, key=x, attention_mask=attn_mask, return_attention_scores=True)
        
    x = keras.layers.LayerNormalization()(keras.layers.Add()([x, attn_out]))
    m_f = ops.expand_dims(ops.cast(mask_1d, "float32"), -1)
    
    latent = ops.sum(x * m_f, axis=1) / ops.maximum(ops.sum(m_f, axis=1), 1.0)
    out = keras.layers.Dense(1, name="AEV_output")(keras.layers.Dense(128, activation='relu')(latent))
    
    p_in = keras.Input(shape=(1,), name="Mass_Param")
    p_s = ops.log1p(p_in / 10.0)
    class_out = keras.layers.Dense(1, activation='sigmoid', name="Prob_Output")(keras.layers.Dense(64, activation='relu')(keras.layers.concatenate([p_s, out])))
    
    inputs = [p_in, vis_inp, phys_inp] if use_kinematic else [p_in, vis_inp]
    encoder_inputs = [vis_inp, phys_inp] if use_kinematic else vis_inp
    
    model = keras.Model(inputs, class_out)
    encoder = keras.Model(encoder_inputs, [out, weights])
    return model, encoder

# --- 5. TRAINING & EXHAUSTIVE PRINT OUTPUTS ---
print("\n" + "="*50)
print("   BUILDING ARCHITECTURES & SCALING DIAGNOSTICS")
print("="*50)

std_m, std_e = build_transformer(False); std_m.compile(optimizer='adam', loss='binary_crossentropy')
#kin_m, kin_e = build_transformer(True); kin_m.compile(optimizer='adam', loss='binary_crossentropy')

print("\n[ ENCODER SUMMARY ]")
std_e.summary()
#kin_e.summary()
print("\n[ FULL CLASSIFIER SUMMARY ]")
#kin_m.summary()
std_m.summary()

try:
    #keras.utils.plot_model(kin_e, to_file=f'{PLOT_DIR}/arch_kin_encoder.png', show_shapes=True, show_layer_names=True)
    #keras.utils.plot_model(kin_m, to_file=f'{PLOT_DIR}/arch_kin_full.png', show_shapes=True, show_layer_names=True)
    keras.utils.plot_model(std_e, to_file=f'{PLOT_DIR}/arch_std_encoder.png', show_shapes=True, show_layer_names=True)
    keras.utils.plot_model(std_m, to_file=f'{PLOT_DIR}/arch_std_full.png', show_shapes=True, show_layer_names=True)
    print("\n[SUCCESS] Architecture diagrams saved.")
except Exception as e:
    print("\n[SKIP] Could not plot model architectures (requires graphviz).")

print("\nTraining Standard Attention (15 Epochs)...")
hist_std = std_m.fit(x=[p_train, vis_train], y=y_train, batch_size=1024, epochs=15, validation_split=0.1, verbose=1)

# --- ADD THIS: SAVE THE MODELS ---
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("\nSaving trained models to disk...")
std_m.save(f"{MODEL_DIR}/MEGAstd_full_classifier.keras")
std_e.save(f"{MODEL_DIR}/MEGAstd_encoder.keras")
print("[SUCCESS] Models saved successfully!")

# --- 6. PERFORMANCE DIAGNOSTICS ---
print("\nFreeing up RAM and VRAM before inference...")
# Delete training arrays to free up system RAM
try:
    del p_train, vis_train, y_train
except NameError:
    pass

import gc
gc.collect()
tf.keras.backend.clear_session() 
print("\nGenerating Analysis Results...")
print("\nGenerating Analysis Results...")
# Dropping batch_size to 64 to bypass VRAM fragmentation
std_out = std_e.predict(test_vis, batch_size=64, verbose=0)
# -----------------------

print("\n DONE training")
# Also dropping batch_size to 512 to protect the V100 GPU's memory from the 72x72 matrices!
#hist_kin = kin_m.fit(x=[p_train, vis_train, phys_train], y=y_train, batch_size=512, epochs=15, validation_split=0.1, verbose=1)
# --- 6. PERFORMANCE DIAGNOSTICS ---

#kin_out = kin_e.predict([test_vis, test_phys], batch_size=1024, verbose=0)

std_scores, std_weights = std_out[0].squeeze(), std_out[1]
#kin_scores, kin_weights = kin_out[0].squeeze(), kin_out[1]

flip_std = scipy.stats.kendalltau(std_scores, test_p.flatten()).correlation < 0
#flip_kin = scipy.stats.kendalltau(kin_scores, test_p.flatten()).correlation < 0
if flip_std: std_scores *= -1
#if flip_kin: kin_scores *= -1

std_tau = scipy.stats.kendalltau(std_scores, test_p.flatten()).correlation
#kin_tau = scipy.stats.kendalltau(kin_scores, test_p.flatten()).correlation

# --- PRINT A COUPLE OF EVENTS ---
print("\n" + "="*50)
print("   INFERENCE VERIFICATION & INPUT MATRIX (FIRST 3 EVENTS)")
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
    #print(f"Kinematic Extracted V : {kin_scores[i]:.4f}")

# --- 7. NEW PLOT: VISUALIZE THE RAW INPUT TENSOR ---
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

# --- 8. ISOLATED VARIABLE MULTIPLICITY CASCADE FEYNMAN DIAGRAM ---
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
    """Dynamically draws a sequential cascade for an A particle into m daughters"""
    step_x = width / (num_daughters - 0.5)
    curr_x, curr_y = sx, sy
    
    for d in range(num_daughters - 1):
        next_x = curr_x + step_x
        if d == num_daughters - 2:
            # Final split into the last two daughters
            draw_line(ax, (curr_x, curr_y), (next_x, curr_y + 0.15), f"$p_{{{A_idx},{d+1}}}$")
            draw_line(ax, (curr_x, curr_y), (next_x, curr_y - 0.15), f"$p_{{{A_idx},{d+2}}}$", text_offset=(0, -0.05))
        else:
            # Emit one daughter up, remnant A continues down and to the right
            next_y_remnant = curr_y - 0.08
            draw_line(ax, (curr_x, curr_y), (next_x, curr_y + 0.15), f"$p_{{{A_idx},{d+1}}}$")
            draw_line(ax, (curr_x, curr_y), (next_x, next_y_remnant), "") # Remnant line (no text)
            curr_x, curr_y = next_x, next_y_remnant

# 1. Main Backbone (Sequential X Decay)
draw_line(ax_feyn, (0, 0.5), (1.5, 0.5), "$X$")
draw_line(ax_feyn, (1.5, 0.5), (3.0, 0.8), "$A_1$")          # First A emitted
draw_line(ax_feyn, (1.5, 0.5), (3.0, 0.2), "$X_{remnant}$")  # X continues

draw_line(ax_feyn, (3.0, 0.2), (4.5, 0.4), "$A_2$")        # Second A emitted
draw_line(ax_feyn, (3.0, 0.2), (4.5, -0.2), "$A_3$")       # Final A emitted

# 2. Add the sequential cascades for each A particle (demonstrating 2-6 variability)
# Notation: p_{1,2} means the 2nd daughter particle of A_1
draw_A_sequential(ax_feyn, 3.0, 0.8, A_idx=1, num_daughters=3, width=2.5)  # A1 -> 3 particles
draw_A_sequential(ax_feyn, 4.5, 0.4, A_idx=2, num_daughters=5, width=3.5)  # A2 -> 5 particles
draw_A_sequential(ax_feyn, 4.5, -0.2, A_idx=3, num_daughters=2, width=1.5) # A3 -> 2 particles

ax_feyn.set_xlim(0, 8.5)
ax_feyn.set_ylim(-0.6, 1.2)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/isolated_feynman_diagram.pdf")

# --- 9. 1D HISTOGRAM COMPARISON (SEPARATED & EXPANDED) ---
print("Generating 1D Histograms for Signal Separation...")
# Expanded mass points
plot_masses = [10, 25, 40, 55, 70, 85]
colors_list = ['purple', 'blue', 'c', 'green', 'orange', 'red']
ls_list = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed']

# Figure 1: Standard Attention (results 1D seperation)
fig_std, ax_std = plt.subplots(figsize=(8, 6))
fig_std.suptitle("results 1D seperation", fontsize=18)

# Figure 2: Kinematic Attention
# fig_kin, ax_kin = plt.subplots(figsize=(8, 6))
# fig_kin.suptitle("1D Signal Separation Profiles (Kinematic Attention)", fontsize=18)

for m, c, ls in zip(plot_masses, colors_list, ls_list):
    #_, v_d, p_d, _ = generate_pair_production_dataset(10000, rng, fixed_mass=m)
    _, v_d, _ = generate_pair_production_dataset(10000, rng, fixed_mass=m)
    std_p_np = std_e.predict(v_d, batch_size=64, verbose=0)[0].squeeze() * (-1 if flip_std else 1)
    #kin_p_np = kin_e.predict([v_d, p_d], batch_size=1024, verbose=0)[0].squeeze() * (-1 if flip_kin else 1)
    
    # ax_std.hist(std_p_np, bins=100, density=True, histtype='step', color=c, linestyle=ls, label=f"$m_A = {m}$ GeV", lw=2.5)
    # ax_kin.hist(kin_p_np, bins=100, density=True, histtype='step', color=c, linestyle=ls, label=f"$m_A = {m}$ GeV", lw=2.5)

# Format Standard Attention Plot
ax_std.set_xlabel("Artificial Variable V"); ax_std.set_ylabel("Density")
ax_std.legend(frameon=False, loc='upper left')
ax_std.grid(alpha=0.2)
fig_std.tight_layout()
fig_std.savefig(f"{PLOT_DIR}/results_1D_seperation.pdf")

# Format Kinematic Attention Plot
# ax_kin.set_xlabel("Artificial Variable V"); ax_kin.set_ylabel("Density")
# ax_kin.legend(frameon=False, loc='upper left')
# ax_kin.grid(alpha=0.2)
# fig_kin.tight_layout()
# fig_kin.savefig(f"{PLOT_DIR}/pairprod_1D_hists_kinematic.pdf")

# --- 10. HEATMAP COMPARISON ---
# --- 10. SEPARATE HEATMAP CORRELATION PLOTS ---
print("Generating Separate 2D Heatmaps...")

# Figure 1: Standard Attention Heatmap
fig_heat_std, ax_heat_std = plt.subplots(figsize=(8, 7))
h_std = ax_heat_std.hist2d(std_scores, test_p.flatten(), bins=80, cmap='viridis')
fig_heat_std.colorbar(h_std[3], ax=ax_heat_std, label='Density')
ax_heat_std.set_title(fr"Standard Attention Matrix" + "\n" + fr"Kendall $\tau = {std_tau:.3f}$", fontsize=14)
ax_heat_std.set_xlabel(r"Artificial Variable $V$")
ax_heat_std.set_ylabel(r"True Resonance Mass $m_A$ (GeV)")
fig_heat_std.tight_layout()
fig_heat_std.savefig(f"{PLOT_DIR}/heatmap_standard.pdf")

# Figure 2: Kinematic Attention Heatmap
# fig_heat_kin, ax_heat_kin = plt.subplots(figsize=(8, 7))
# h_kin = ax_heat_kin.hist2d(kin_scores, test_p.flatten(), bins=80, cmap='viridis')
# fig_heat_kin.colorbar(h_kin[3], ax=ax_heat_kin, label='Density')
# ax_heat_kin.set_title(fr"Kinematic Attention Matrix" + "\n" + fr"Kendall $\tau = {kin_tau:.3f}$", fontsize=14)
# ax_heat_kin.set_xlabel(r"Artificial Variable $V$")
# ax_heat_kin.set_ylabel(r"True Resonance Mass $m_A$ (GeV)")
# fig_heat_kin.tight_layout()
# fig_heat_kin.savefig(f"{PLOT_DIR}/heatmap_kinematic.pdf")
# --- 11. ATTENTION MAPS ---
print("Generating Attention Maps...")
evt_idx = np.random.randint(0, len(test_vis))



# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# fig.suptitle(f"Attention Weight Allocation (Event {evt_idx})", fontsize=16)
# im0 = axes[0].imshow(np.mean(std_weights[evt_idx], axis=0), cmap='hot', interpolation='nearest', vmin=0, vmax=0.3)
# axes[0].set_title("Standard Attention Matrix"); axes[0].set_xlabel("Key Particle"); axes[0].set_ylabel("Query Particle")
# fig.colorbar(im0, ax=axes[0])

# # im1 = axes[1].imshow(np.mean(kin_weights[evt_idx], axis=0), cmap='hot', interpolation='nearest', vmin=0, vmax=0.3)
# # axes[1].set_title("Kinematic Attention Matrix"); axes[1].set_xlabel("Key Particle")
# # fig.colorbar(im1, ax=axes[1])

# for ax in axes: 
#     ax.set_xticks(np.arange(0, MAX_PARTICLES, 8))
#     ax.set_yticks(np.arange(0, MAX_PARTICLES, 8))
# plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/pairprod_attention.pdf")



fig, ax = plt.subplots(1, 1, figsize=(7, 6))
fig.suptitle(f"Attention Weight Allocation (Event {evt_idx})", fontsize=16)

im0 = ax.imshow(np.mean(std_weights[evt_idx], axis=0), cmap='hot', interpolation='nearest', vmin=0, vmax=0.3)
ax.set_title("Standard Attention Matrix"); ax.set_xlabel("Key Particle"); ax.set_ylabel("Query Particle")
fig.colorbar(im0, ax=ax)

ax.set_xticks(np.arange(0, MAX_PARTICLES, 8))
ax.set_yticks(np.arange(0, MAX_PARTICLES, 8))
# ==============================================================================
# --- 12. ADVANCED EVALUATION PLOTS (LOSS, ROC, PROFILE) ---
# ==============================================================================
print("Generating Loss, ROC, and Profile Plots...")

# --- A. SEPARATE LOSS CURVES ---
print("Generating Separate Loss Curves...")

# Figure 1: Standard Attention Loss

fig_loss_std, ax_loss_std = plt.subplots(figsize=(8, 6))
ax_loss_std.plot(hist_std.history['loss'], label='Train Loss', color='blue', lw=2)
ax_loss_std.plot(hist_std.history['val_loss'], label='Val Loss', color='lightblue', lw=2, linestyle='dashed')
ax_loss_std.set_title("Standard Attention: Training and Validation Loss", fontsize=16)
ax_loss_std.set_xlabel("Epoch", fontsize=14)
ax_loss_std.set_ylabel("Binary Crossentropy Loss", fontsize=14)
ax_loss_std.legend(fontsize=12)
ax_loss_std.grid(alpha=0.3)
fig_loss_std.tight_layout()
fig_loss_std.savefig(f"{PLOT_DIR}/loss_curve_standard.pdf")

# Figure 2: Kinematic Attention Loss
# fig_loss_kin, ax_loss_kin = plt.subplots(figsize=(8, 6))
# ax_loss_kin.plot(hist_kin.history['loss'], label='Train Loss', color='green', lw=2)
# ax_loss_kin.plot(hist_kin.history['val_loss'], label='Val Loss', color='lightgreen', lw=2, linestyle='dashed')
# ax_loss_kin.set_title("Kinematic Attention: Training and Validation Loss", fontsize=16)
# ax_loss_kin.set_xlabel("Epoch", fontsize=14)
# ax_loss_kin.set_ylabel("Binary Crossentropy Loss", fontsize=14)
# ax_loss_kin.legend(fontsize=12)
# ax_loss_kin.grid(alpha=0.3)
# fig_loss_kin.tight_layout()
# fig_loss_kin.savefig(f"{PLOT_DIR}/loss_curve_kinematic.pdf")
# --- B. PARAMETERIZED ROC CURVE ---
# To test the classifier, we feed it True Mass pairings (Label 1) and Fake Mass pairings (Label 0)
roc_p_true = test_p
roc_p_fake = rng.uniform(M0_range[0], M0_range[1], test_p.shape)

roc_p = np.concatenate([roc_p_true, roc_p_fake])
roc_vis = np.concatenate([test_vis, test_vis])
# roc_phys = np.concatenate([test_phys, test_phys])
roc_y = np.concatenate([np.ones(len(test_p)), np.zeros(len(test_p))])

# Predict probabilities from the FULL classifier models
pred_std = std_m.predict([roc_p, roc_vis], batch_size=64, verbose=0).squeeze()
#pred_kin = kin_m.predict([roc_p, roc_vis, roc_phys], batch_size=1024, verbose=0).squeeze()

fpr_std, tpr_std, _ = roc_curve(roc_y, pred_std)
auc_std = auc(fpr_std, tpr_std)

#fpr_kin, tpr_kin, _ = roc_curve(roc_y, pred_kin)
#auc_kin = auc(fpr_kin, tpr_kin)

fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
ax_roc.plot(fpr_std, tpr_std, color='blue', lw=2, label=f'Standard Attention (AUC = {auc_std:.4f})')
#ax_roc.plot(fpr_kin, tpr_kin, color='green', lw=2, label=f'Kinematic Attention (AUC = {auc_kin:.4f})')
ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (Fake Mass Assignment)', fontsize=14)
ax_roc.set_ylabel('True Positive Rate (True Mass Assignment)', fontsize=14)
ax_roc.set_title('Classifier ROC Curve (Parameterized Background vs Signal)', fontsize=16)
ax_roc.legend(loc="lower right", fontsize=12)
ax_roc.grid(alpha=0.3)




plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/roc_curves.pdf")



# --- ADD THIS RIGHT AFTER SAVING THE ROC CURVES (in Section 12) ---
print("Generating Classifier Score Distributions...")

# Separate the predictions based on the true labels
scores_class1 = pred_std[roc_y == 1]  # True Mass Pairings
scores_class0 = pred_std[roc_y == 0]  # Fake Mass Pairings

fig_score, ax_score = plt.subplots(figsize=(8, 6))

# Plot overlapping histograms
ax_score.hist(scores_class1, bins=50, alpha=0.5, color='blue', label='True Pairings (Class 1)', density=True)
ax_score.hist(scores_class0, bins=50, alpha=0.5, color='red', label='Fake Pairings (Class 0)', density=True)

# Add outline/step for better visibility
ax_score.hist(scores_class1, bins=50, color='blue', histtype='step', lw=1.5, density=True)
ax_score.hist(scores_class0, bins=50, color='red', histtype='step', lw=1.5, density=True)

ax_score.set_title("Classifier Output Scores (Standard Attention)", fontsize=16)
ax_score.set_xlabel("Predicted Probability (Sigmoid Output)", fontsize=14)
ax_score.set_ylabel("Density", fontsize=14)
ax_score.legend(loc='upper center', fontsize=12)
ax_score.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/classifier_scores_distribution.pdf")
# --- C. SEPARATE PROFILE CORRELATION PLOTS ---
# --- C. SEPARATE PROFILE CORRELATION PLOTS ---
print("Generating Separate Profile Plots...")
# Calculates the Mean and Std Dev of V in discrete mass bins
bins = np.linspace(5, 100, 20)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

mean_std, bin_edges, _ = scipy.stats.binned_statistic(test_p.flatten(), std_scores, statistic='mean', bins=bins)
std_dev_std, _, _ = scipy.stats.binned_statistic(test_p.flatten(), std_scores, statistic='std', bins=bins)

# mean_kin, _, _ = scipy.stats.binned_statistic(test_p.flatten(), kin_scores, statistic='mean', bins=bins)
# std_dev_kin, _, _ = scipy.stats.binned_statistic(test_p.flatten(), kin_scores, statistic='std', bins=bins)

# Figure 1: Standard Attention Profile
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

# Figure 2: Kinematic Attention Profile
# fig_prof_kin, ax_prof_kin = plt.subplots(figsize=(8, 6))
# ax_prof_kin.errorbar(bin_centers, mean_kin, yerr=std_dev_kin, fmt='-s', color='green', capsize=5, 
#                  label=f'Kinematic Attn ($\\tau$={kin_tau:.3f})', alpha=0.8)
# ax_prof_kin.set_xlabel(r"True Resonance Mass $m_A$ (GeV)", fontsize=14)
# ax_prof_kin.set_ylabel(r"Mean Extracted Variable $\langle V \rangle$", fontsize=14)
# ax_prof_kin.set_title("Kinematic Attention: Calibration Profile", fontsize=16)
# ax_prof_kin.legend(loc='upper left', fontsize=12)
# ax_prof_kin.grid(alpha=0.3)
# fig_prof_kin.tight_layout()
# fig_prof_kin.savefig(f"{PLOT_DIR}/profile_kinematic.pdf")


# ==============================================================================
# --- 13. MASS RECONSTRUCTION (CALIBRATION TO PHYSICAL UNITS) ---
# ==============================================================================
print("Calibrating V to Reconstructed Mass (GeV)...")

# 1. Fit a 3rd-degree polynomial to map V -> m_A using the continuous test set
# poly_kin = np.polyfit(kin_scores, test_p.flatten(), deg=3)
# calib_kin = np.poly1d(poly_kin)

poly_std = np.polyfit(std_scores, test_p.flatten(), deg=3)
calib_std = np.poly1d(poly_std)

# Print the calibration equations to the console
print("\n--- CALIBRATION EQUATIONS ---")
print(f"Standard Attn : m_reco = {poly_std[0]:.4f}*V^3 + {poly_std[1]:.4f}*V^2 + {poly_std[2]:.4f}*V + {poly_std[3]:.4f}")
# print(f"Kinematic Attn: m_reco = {poly_kin[0]:.4f}*V^3 + {poly_kin[1]:.4f}*V^2 + {poly_kin[2]:.4f}*V + {poly_kin[3]:.4f}")

# 2. Generate Reconstructed Mass Spectra Plots
#fig_reco, axes_reco = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig_reco, ax_reco = plt.subplots(1, 1, figsize=(8, 6))
fig_reco.suptitle("Reconstructed Invariant Mass Spectra (Mapped to GeV)", fontsize=18)

plot_masses = [20, 40, 60, 80]
colors_list = ['orange', 'red', 'blue', 'green']

for m, c in zip(plot_masses, colors_list):
    # Generate pure signal for specific hidden masses
    #_, v_d, p_d, _ = generate_pair_production_dataset(10000, rng, fixed_mass=m)
    _, v_d, _ = generate_pair_production_dataset(10000, rng, fixed_mass=m)
    # Predict raw V
    raw_v_std = std_e.predict(v_d, batch_size=64, verbose=0)[0].squeeze() * (-1 if flip_std else 1)
    # raw_v_kin = kin_e.predict([v_d, p_d], batch_size=1024, verbose=0)[0].squeeze() * (-1 if flip_kin else 1)
    
    # Apply Calibration to convert V to GeV
    reco_mass_std = calib_std(raw_v_std)
    # reco_mass_kin = calib_kin(raw_v_kin)
    
    # Plot the physical reconstructed masses
    # Plot the physical reconstructed masses
    ax_reco.hist(reco_mass_std, bins=80, histtype='step', color=c, label=f"True $m_A$ = {m} GeV", lw=2.5)
    
    # Add a vertical dotted line showing exactly where the TRUE mass is
    ax_reco.axvline(m, color=c, linestyle=':', alpha=0.6, lw=2)
    # axes_reco[1].axvline(m, color=c, linestyle=':', alpha=0.6, lw=2)
    # axes_reco[0].hist(reco_mass_std, bins=80, density=True, histtype='step', color=c, label=f"True $m_A$ = {m} GeV", lw=2.5)
    # # axes_reco[1].hist(reco_mass_kin, bins=80, density=True, histtype='step', color=c, label=f"True $m_A$ = {m} GeV", lw=2.5)
    
    # # Add a vertical dotted line showing exactly where the TRUE mass is
    # axes_reco[0].axvline(m, color=c, linestyle=':', alpha=0.6, lw=2)
    # axes_reco[1].axvline(m, color=c, linestyle=':', alpha=0.6, lw=2)

# Format Standard Plot
# Format Standard Plot
ax_reco.set_title("Standard Attention")
ax_reco.set_xlabel(r"Reconstructed Mass $m_{reco}$ (GeV)", fontsize=14)
ax_reco.set_ylabel("Density", fontsize=14)
ax_reco.legend(frameon=False, loc='upper left')
ax_reco.grid(alpha=0.2)
ax_reco.set_xlim([0, 100])

# Format Kinematic Plot
# axes_reco[1].set_title("Kinematic Attention")
# axes_reco[1].set_xlabel(r"Reconstructed Mass $m_{reco}$ (GeV)", fontsize=14)
# axes_reco[1].legend(frameon=False, loc='upper left')
# axes_reco[1].grid(alpha=0.2)
# axes_reco[1].set_xlim([0, 100])

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/reconstructed_mass_spectra.pdf")
