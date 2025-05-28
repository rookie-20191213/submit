# GPRImagingPy

## Getting Started

### What is GPRImagingPy?

**GPRImagingPy** is an open-source software package for migration and inversion imaging of both synthetic and field Ground Penetrating Radar (GPR) data. It integrates advanced imaging techniques, including:

- **F-K Migration**
- **Reverse Time Migration (RTM)** based on the **exploding reflector model** and **cross-correlation imaging condition**
- **Full Waveform Inversion (FWI)** with regularization

These methods are implemented to provide robust and accurate imaging solutions for a wide range of GPR applications.

**GPRImagingPy** supports complete workflows for both synthetic and field GPR datasets.

### License

This project is released under the [MIT License](https://opensource.org/licenses/MIT), allowing free use, modification, and distribution.

### Implementation

- Written in **pure Python 3**
- Performance-critical components are accelerated using **Numba**
- Supports **multi-processing** for parallel computation and improved efficiency

## Package Overview

The software is organized into several example modules, each corresponding to a key component of the GPR imaging workflow. Below is a description of the main directories and their associated files.

---

### `/00ForwardModeling`

This module contains scripts for forward modeling of GPR wave propagation.

**File Descriptions:**

- `add_cpml.py` – Implements CPML (Convolutional Perfectly Matched Layer) absorbing boundary conditions.
- `clutter_removal.py` – Provides methods for extracting air-ground coupled waves.
- `config_parameters.py` – Defines and stores model configuration parameters.
- `demo.py` – A demonstration script showcasing forward modeling.
- `display.py` – Visualization tools for forward modeling results.
- `forward_modeling.py` – Core class for forward wavefield modeling.
- `model_creation.py` – Utility for building physical subsurface models.
- `update_field_forward.py` – Updates electric and magnetic field components in the forward wavefield.
- `wavelet_creation.py` – Constructs various types of source wavelets.

---

### `/01Migration`

This module includes migration imaging algorithms.

#### `/fk_migration`

- `fk_migration.py` – Implements the F-K migration method.

#### `/reverse_time_migration`

- `add_cpml.py` – Implements CPML absorbing boundaries.
- `clutter_removal.py` – Extracts air-ground coupled wave interference.
- `config_parameters.py` – Stores model configuration parameters.
- `demo.py` – Demonstration script for reverse time migration (RTM).
- `imaging_condition.py` – Implements different imaging conditions.
- `update_field_forward.py` – Updates electric and magnetic fields for the source wavefield.
- `update_field_reverse.py` – Updates electric and magnetic fields for the receiver wavefield.
- `wavelet_creation.py` – Generates various source wavelets.

---

### `/03FullWaveformInversion`

This module contains the full waveform inversion (FWI) workflow with regularization.

**File Descriptions:**

- `add_cpml.py` – Adds CPML absorbing boundary layers.
- `clutter_removal.py` – Handles air-ground coupled wave suppression.
- `config_parameters.py` – Configures and stores model parameters.
- `creat_model.py` – Constructs the physical subsurface model.
- `demo.py` – Example script demonstrating FWI.
- `display.py` – Visualizes inversion and modeling results.
- `forward_modeling.py` – Performs forward wave propagation simulations.
- `fwi_parameters.py` – Stores inversion-specific parameters.
- `gradient_calculation.py` – Computes gradients for guiding the inversion process.
- `optimization.py` – Optimization algorithms for FWI.
- `regularization_method.py` – Regularization techniques used in FWI.
- `update_field_adjoint.py` – Updates electric and magnetic fields for the adjoint wavefield.
- `update_field_forward.py` – Updates fields for the forward wavefield.
- `wavelet_creation.py` – Defines excitation wavelets for different scenarios.

---

Each component is modular and extensible, making **GPRImagingPy** a flexible platform for both academic research and applied geophysical imaging tasks.

## 🛠️ Installation

The following steps guide you through how to install **GPRImagingPy**:

### 📌 Step 1: Install Python and Required Packages

Ensure that you have **Python 3** installed. Then, install the necessary Python libraries by running:

```bash
pip install numpy numba matplotlib scikit-image
```

> **Note**: The `multiprocessing` module is part of the Python standard library and does not require separate installation.

---

### 🔽 Step 2: Clone the Source Code from GitHub

You can retrieve the source code by running the following command:

```bash
git clone https://github.com/rookie-20191213/submit.git
```

---

### 🧰 Software Requirements

- **External Software**: None
- **Python Dependencies**:
  - `numpy`
  - `numba`
  - `matplotlib`
  - `scikit-image`
  - `multiprocessing` (built-in)

---

### 💻 Platform Requirement

- ✅ Tested on: **Ubuntu 24.04.1 LTS**
- ❌ Not supported on: **Windows**

> ⚠️ **Important Notice for Windows Users**:  
> This software is not executable on Windows due to platform-specific multiprocessing constraints.  
> In Windows, the `multiprocessing` module uses the **spawn** method to start new processes. This requires **all multiprocessing logic to be enclosed within the following structure**:

```python
if __name__ == "__main__":
    # your multiprocessing code here
```

> Without this structure, the program may enter unintended recursive spawning of processes and fail to run correctly.  
> As a result, **GPRImagingPy is currently not supported on Windows** systems.

---

Once installed, you are ready to begin modeling, migration, and full waveform inversion with **GPRImagingPy**.
