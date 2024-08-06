# Expected value Constrained Stochastic Compositional Gradient Descent (EC-SCGD) algorithm

This project involves running Julia implementation of stochastic gradient descent methods for the stochastic compositional optimization with compositional constraints.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Installing Julia](#installing-julia)
  - [Installing Julia Packages](#installing-julia-packages)
- [Usage](#usage)
  - [Running the Main File](#running-the-main-file)
- [Files](#files)

## Installation

### Installing Julia

1. **Download Julia**:
   - Go to the [Julia website](https://julialang.org/downloads/) and download the appropriate version for your operating system.

2. **Install Julia**:
   - Follow the installation instructions for your operating system:
     - **Windows**: Run the installer and follow the prompts.
     - **macOS**: Open the downloaded `.dmg` file and drag the Julia app to your Applications folder.
     - **Linux**: Extract the downloaded tarball and move the extracted folder to `/opt`. Then create a symbolic link to the `julia` executable in `/usr/local/bin`.

3. **Verify Installation**:
   - Open a terminal or command prompt and type `julia` to start the Julia REPL.
   - Type `versioninfo()` in the REPL to verify the installation.

### Installing Julia Packages

1. **Open Julia REPL**:
   - Start the Julia REPL by typing `julia` in your terminal or command prompt.

2. **Install Required Packages**:
   - Use the package manager to install the necessary packages. In the Julia REPL, type:
     ```julia
     using Pkg
     Pkg.add("PackageName")
     ```
   - The required packages for this project include:
     - `LinearAlgebra`
     - `Random`
     - `Statistics`
     - `Plots`
     - `PyPlot`
     - `NPZ`

   - To install all required packages at once, you can use:
     ```julia
     Pkg.add(["LinearAlgebra", "Random", "Statistics", "Plots", "PyPlot", "NPZ"])
     ```

## Usage

### Running the Main File

1. **Navigate to the Project Directory**:
   - Open a terminal or command prompt and navigate to the directory where the project files are located.

2. **Run the Main File**:
   - In the terminal or command prompt, execute the following command:
     ```sh
     julia run.jl
     ```

   - This will run the `run.jl` script, which is the main entry point for this project.

## Files

- `run.jl`: The main script to run the project.
- `Utils.jl`: Utility functions used in the project.
- `EC_SCGD.jl`: Implementation of the Expected value Constrained Stochastic Compositional Gradient Descent algorithm.
- `gradient_descent_solvers.jl`: Gradient descent solvers for the oracle problems used in the project.

Ensure all these files are in the same directory when running the `run.jl` script.
