# LatticeFlow: GNN Surrogate & RL Optimization for Crashworthiness

LatticeFlow is a two-stage machine learning framework designed to accelerate the design and optimization of crash-absorbing Voronoi structures. Traditional crashworthiness design relies on Finite Element Analysis (FEA), which is computationally expensive and slow. LatticeFlow bypasses this bottleneck by replacing the FEA solver with a **Graph Neural Network (GNN)** surrogate, which is then used as a high-speed physics environment for a **Reinforcement Learning (RL)** agent to optimize the structure's geometry.

Currently optimized for 2D quasistatic crush of Thermoplastic Polyurethane (TPU) Voronoi absorbers, this pipeline achieves design iterations in seconds rather than hours.
This project validated this framework by optimizing the seed point locations of Voronoi Structure, these seed points are the primary design variable that defines the geometry of the structure. Hence the RL agent used this as the primary optimizing variable. 
      
      Status:The Project is undergoing improvements specifically to the GNN (MGN) Model. 
             This framework can be adpated to any problems involving discretization of domains, however the currenly the model supports Quasi-static loads and 2D geomentries.


      NOTE: Voronoi structure is a geometric pattern created by scattering points (called "seeds") across a space and drawing borders exactly halfway between them.
      The defining rule is simple: every location inside a specific cell is closer to its own seed than to any other seed.
      In nature, you see this pattern in soap bubbles, leaf veins, and giraffe spots. In engineering, these irregular, foam-like lattices are heavily used to design                materials that are lightweight but exceptionally good at absorbing energy and distributing stress. 

---

## The Workflow

The pipeline is split into two distinct stages:

<p align="center">
  <img src="assets/pipeline.png" alt="LatticeFlow Pipeline" width="900"/>
</p>

### Stage 1: The Physics Surrogate (MeshGraphNet)
Instead of solving differential equations, we train a MeshGraphNet (MGN) to understand the physical relationships between structural components. The model treats the FEA mesh as a graph, where nodes are physical coordinates and edges are the structural bonds (or collision contacts).

* **Architecture & Node Encoding**: The network operates on a heterogeneous graph containing Mesh edges (structural connections) and World edges (dynamic contact points between opposing surfaces). Each node $i$ captures velocity history, mass, and categorical flags (e.g., impactor, absorber, fixed boundary):

$$\mathbf{x}_i^t = \text{MLP}_{\text{enc}}\left(\left[\ \mathbf{v}_i^{t-1},\ \mathbf{v}_i^{t},\ m_i^{-1},\ \mathbf{f}_i\ \right]\right)$$

* **Gated Message Passing**: A common issue in deep GNNs is "over-smoothing," where node features become indistinguishable after many layers. To preserve localized physical phenomena (like buckling or sharp impacts) across 15 interaction layers, LatticeFlow utilizes a gated update mechanism. The gate acts as a valve, learning exactly how much new information to blend into the existing state:

$$\mathbf{c} = \text{MLP}_{\text{update}}(\mathbf{x}_{\text{in}}), \quad g = \sigma\left(\text{MLP}_{\text{gate}}(\mathbf{x}_{\text{in}})\right)$$

$$\mathbf{h}^{\ell+1} = (1 - g) \odot \mathbf{h}^{\ell} + g \odot \mathbf{c}$$

<p align="center">
  <img src="assets/MGN_architecture.png" alt="Architecture" width="450"/>
  <img src="assets/gating_mechanism.png" alt="Gating" width="350"/>
</p>

* **Edge Kinematics & Objective**: To accurately predict deformation, the mesh edges are encoded with both geometric distances and a dedicated strain vector $\boldsymbol{\varepsilon}_{ij}$ tracking compression. The surrogate is penalized for node-level acceleration errors and relative structural inconsistencies across the mesh and world edges:

$$\mathbf{e}_{ij}^M = \left[\ \mathbf{p}_i - \mathbf{p}_j,\ \boldsymbol{\varepsilon}_{ij},\ \mathbf{v}_i - \mathbf{v}_j,\ \lVert\mathbf{p}_i - \mathbf{p}_j\rVert,\ \lVert\mathbf{p}_i^0 - \mathbf{p}_j^0\rVert\ \right]$$

$$\mathcal{L} = \frac{1}{\lvert\mathcal{A}\rvert}\sum_{i \in \mathcal{A}} \left(\lvert\hat{\mathbf{a}}_i - \mathbf{a}_i\rvert + 0.5\,\mathcal{L}_{\text{mesh},i} + 0.5\,\mathcal{L}_{\text{world},i}\right)$$

### Stage 2: Structural Optimization (Phasic PPO)
With the surrogate trained, we deploy a Phasic Policy Gradient (PPG) agent to act as the designer.

* **Problem Formulation**: The internal lattice structure is dictated by a set of Voronoi seed points. The RL agent observes the normalized coordinates and topological relationships of these seeds, and outputs displacement actions. The ultimate reward is the improvement in **Crush Force Efficiency (CFE)**:

$$r_t = \text{CFE}(\mathbf{p}_{t+1}) - \text{CFE}(\mathbf{p}_t)$$

* **Squashed Gaussian Policy**: Because seed coordinates must remain within a physical bounding box, we sample actions from an unbounded latent space and compress them using a $\tanh$ function. This requires a change-of-variables correction to calculate the true probability density:

$$\log \pi_\theta(\mathbf{a} \mid \mathbf{s}) = \sum_{d=1}^{2K} \left[\log \mathcal{N}(u_d \mid \mu_d, \sigma_d^2) - \log(1 - \tanh^2(u_d)) - \log(a_{\max})\right]$$

* **Phasic Updates for Stability**: Standard PPO updates the policy and value functions simultaneously, which can destabilize learning when evaluating highly sensitive geometric structures. LatticeFlow decouples this into phases: fitting the critic, refining advantages, and updating the policy with a corrected entropy bonus that accounts for the $\tanh$ action-space compression:

$$\mathcal{L}_\pi = -\frac{1}{\lvert\mathcal{B}\rvert}\sum_{i \in \mathcal{B}} \min\left(r_i \hat{A}_i^{\text{norm}},\ \text{clamp}(r_i, 1{-}\epsilon, 1{+}\epsilon)\ \hat{A}_i^{\text{norm}}\right) - c_H \cdot \bar{H}_a[\pi_\theta]$$

---

## Results & Simulations

### Physics Prediction
The surrogate model successfully predicts hundreds of frames of deformation from a single initial state autoregressively.

<p align="center">
  <img src="assets/rollout_animation.gif" alt="Crash Rollout" width="750"/>
</p>

This closely tracks the ground truth reaction forces extracted from traditional FEA solvers:

<p align="center">
  <img src="assets/reaction_force_for_rollout_animation_comparison.png" alt="Ground Truth Reaction Force" width="850"/>
</p>

### Optimized Design
During training, the PPG agent learns to strategically distribute the Voronoi seeds to eliminate dangerous force spikes. By spreading out the load paths, the structure absorbs energy much more smoothly.

<p align="center">
  <img src="assets/PPG_optimization_5_seeds.gif" alt="Optimization" width="650"/>
  <img src="assets/cfe_optimized_force_curve.png" alt="Force Curve" width="650"/>
</p>

---

## Technical Foundations
* **Material Science**: The mechanics in both the physics surrogate and the RL reward rely on an incompressible **Ogden hyperelastic** formulation, explicitly calibrated for Thermoplastic Polyurethane (TPU). The 2D plane-stress strain energy density evaluates how the TPU elements stretch and compress:

$$W(\lambda_1, \lambda_2) = \sum_{p=1}^{2} \frac{\mu_p}{\alpha_p}\left(\lambda_1^{\alpha_p} + \lambda_2^{\alpha_p} + (\lambda_1 \lambda_2)^{-\alpha_p} - 3\right)$$

From this integrated strain energy ($U$), we differentiate to find the reaction force and define the core objective metric:

$$F(y) = -\frac{dU}{dy}, \qquad \text{CFE} = \frac{\text{mean}(|F|)}{\text{peak}(|F|)}$$

* **Data Pipeline**: Raw simulations are exported from COMSOL in Nastran (`.nas`) format alongside text-based trajectory arrays. The preprocessing tools in `/tools` map these files via nearest-neighbor KD-trees, compute inverse nodal masses based on TPU density, build bidirectional graph edges, and save independent `.pt` dictionaries for PyTorch Geometric to ingest.

## Project Layout
* `/src`: Core model logic, including the Gated MeshGraphNet and custom loss functions.
* `/scripts`: Training routines for the surrogate and the PPG optimizer.
* `/tools`: Visualization tools, data preprocessors, and the Voronoi mesh generator.
* `/config`: Centralized YAML files to manage hyperparameters.


## Getting Started

Prerequisites: A CUDA-enabled GPU (8GB+ VRAM) and system memory of 16GB+ is recommended. PyTorch Geometric must be compiled to match your local CUDA version.

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Train the Surrogate Model**:
```bash
python MGN_main.py --config config/MGN_config.yaml --mode train
```

3. **Run Design Optimization**:
```bash
python PPG_main.py --config config/PPG_config.yaml --mode train
```

## Collaboration & Credits
This framework is part of ongoing collaborative research into AI-driven mechanical design. 
Special thanks to [atomic-coder](https://github.com/atomic-coder) for the collaboration on the core MGN architecture.

## Citation
If you utilize this framework or code structure in your own research, please cite:

```bibtex
@software{latticeflow2026,
  author    = {Naveen Kumar},
  title     = {LatticeFlow: Graph Neural Network Surrogate and RL Optimization for Crashworthiness Design},
  year      = {2026},
  url       = {[https://github.com/Naveen15-4/LatticeFlow](https://github.com/Naveen15-4/LatticeFlow)}
}
```

## License
MIT License.
