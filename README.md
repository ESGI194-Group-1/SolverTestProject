SolverTestProject
================

## How to run the code ?

### Julia and package installation
- [Install Julia](https://julialang.org/downloads/)
- Clone the repository with git, cd to the directory
- The repository contains a file 'Project.toml' which lists the Julia packages needed to run the code
  Install them by instantiating this so called project "environment":
```
shell> cd SolverTestProject
shell> julia --project
julia> using Pkg
julia> Pkg.instantiate()
```
This will download and install all needed packages.

### Running the notebook(s)
The `notebooks` folder contains a couple of [Pluto notebooks](https://plutojl.org/). Run them in
your browser as follows:
```
shell> cd SolverTestProject
shell> julia --project
julia> using Pluto
julia> Pluto.run(notebook="notebooks/TrySimple.jl")
```

This will open a new browser tab with the notebook running. Be aware the Julia alway compiles code
to your particular machine when the code is first run after a change. Depending on the code this
make take some time. Repeated runs then will be fast.

