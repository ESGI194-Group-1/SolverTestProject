### A Pluto.jl notebook ###
# v1.0.0

using Markdown
using InteractiveUtils

# ╔═╡ 5967c40d-e4d9-434c-99f7-c577cad5a4de
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using SDDP
    
end

# ╔═╡ c4bb3bd4-7bb7-4e80-ad8f-b36a6c876132
using HiGHS

# ╔═╡ 95f34209-bac3-4e68-a6cc-1874f7c3583a
md"""
# SDDP First Steps
"""

# ╔═╡ d8f5ada3-1e60-4f47-a65a-69e9df0d819c
md"""
This notebook implements code from [https://sddp.dev/stable/tutorial/first_steps/](https://sddp.dev/stable/tutorial/first_steps/)
"""

# ╔═╡ d799f3c8-4500-49d3-80e7-ad896cbf1ee6
graph = SDDP.LinearGraph(3)

# ╔═╡ eb893302-0d9b-4e66-9986-1638fa6e37cc
function subproblem_builder(subproblem::Model, node::Int)
    # State variables
    @variable(subproblem, 0 <= volume <= 200, SDDP.State, initial_value = 200)
    # Control variables
    @variables(subproblem, begin
        thermal_generation >= 0
        hydro_generation >= 0
        hydro_spill >= 0
    end)
    # Random variables
    @variable(subproblem, inflow)
    Ω = [0.0, 50.0, 100.0]
    P = [1 / 3, 1 / 3, 1 / 3]
    SDDP.parameterize(subproblem, Ω, P) do ω
        fix(inflow, ω)
        return
    end
    # Transition function and constraints
    @constraints(
        subproblem,
        begin
            volume.out == volume.in - hydro_generation - hydro_spill + inflow
            demand_constraint, hydro_generation + thermal_generation == 150
        end
    )
    # Stage-objective
    fuel_cost = [50, 100, 150]
    @stageobjective(subproblem, fuel_cost[node] * thermal_generation)
    return subproblem
end

# ╔═╡ 3cd7ba24-8bc2-4dbf-bd0b-22af607fd8c9
model = SDDP.LinearPolicyGraph(
    subproblem_builder;
    stages = 3,
    sense = :Min,
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
)

# ╔═╡ 0b470088-af4f-44bf-9500-7e246277ef4e
SDDP.train(model; iteration_limit = 10)

# ╔═╡ e762a43c-67de-4314-8c6d-551e3498fd11
rule = SDDP.DecisionRule(model; node = 1)

# ╔═╡ 192b204b-9a4d-453d-b7ee-93f51d9d0afa
solution = SDDP.evaluate(
    rule;
    incoming_state = Dict(:volume => 150.0),
    noise = 50.0,
    controls_to_record = [:hydro_generation, :thermal_generation],
)

# ╔═╡ 96eb70c1-940e-44ec-a98b-89c29ba2e2ee
simulations = SDDP.simulate(
    # The trained model to simulate.
    model,
    # The number of replications.
    100,
    # A list of names to record the values of.
    [:volume, :thermal_generation, :hydro_generation, :hydro_spill],
)



# ╔═╡ d9485a75-e6b4-4527-b02d-7a68840db027
replication = 1

# ╔═╡ 669aad52-dc37-456b-85bd-b2e8be445c50
stage = 2

# ╔═╡ adaf379f-81c9-47d3-93f2-acea1002cd67
simulations[replication][stage]

# ╔═╡ 2656e84d-e05a-4d19-a0ef-57159d0b74e0
outgoing_volume = map(simulations[1]) do node
    return node[:volume].out
end

# ╔═╡ 19338d41-35d2-42cd-a294-7eba3248022b
thermal_generation = map(simulations[1]) do node
    return node[:thermal_generation]
end

# ╔═╡ 3558803d-f591-46cd-977f-51c420b957b2
objectives = map(simulations) do simulation
    return sum(stage[:stage_objective] for stage in simulation)
end

# ╔═╡ 7ece512f-3ae4-4ba4-a49d-e4ab9870f53d
μ, ci = SDDP.confidence_interval(objectives)

# ╔═╡ c59c8295-6c70-42ef-a48c-aa111bac8727
println("Lower bound: ", SDDP.calculate_bound(model))

# ╔═╡ fa5a50b8-aa1b-49b2-9823-eaf332268b75
simulations2 = SDDP.simulate(
    model,
    1;  ## Perform a single simulation
    custom_recorders = Dict{Symbol,Function}(
        :price => (sp::Model) -> dual(sp[:demand_constraint]),
    ),
)

# ╔═╡ 41e879d7-9d5c-438f-bb63-09b3c9b19d82
prices = map(simulations2[1]) do node
    return node[:price]
end

# ╔═╡ c2c05ba4-7344-417d-a075-07dbfc3cc9a1
V = SDDP.ValueFunction(model; node = 1)

# ╔═╡ 1bc1ba05-7c01-4ab2-87ca-6444a634bfe7
cost, price = SDDP.evaluate(V, Dict("volume" => 10))

# ╔═╡ 784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
html"""<style>.dont-panic{ display: none }</style>"""


# ╔═╡ Cell order:
# ╠═5967c40d-e4d9-434c-99f7-c577cad5a4de
# ╟─95f34209-bac3-4e68-a6cc-1874f7c3583a
# ╠═d8f5ada3-1e60-4f47-a65a-69e9df0d819c
# ╠═d799f3c8-4500-49d3-80e7-ad896cbf1ee6
# ╠═eb893302-0d9b-4e66-9986-1638fa6e37cc
# ╠═c4bb3bd4-7bb7-4e80-ad8f-b36a6c876132
# ╠═3cd7ba24-8bc2-4dbf-bd0b-22af607fd8c9
# ╠═0b470088-af4f-44bf-9500-7e246277ef4e
# ╠═e762a43c-67de-4314-8c6d-551e3498fd11
# ╠═192b204b-9a4d-453d-b7ee-93f51d9d0afa
# ╠═96eb70c1-940e-44ec-a98b-89c29ba2e2ee
# ╠═d9485a75-e6b4-4527-b02d-7a68840db027
# ╠═669aad52-dc37-456b-85bd-b2e8be445c50
# ╠═adaf379f-81c9-47d3-93f2-acea1002cd67
# ╠═2656e84d-e05a-4d19-a0ef-57159d0b74e0
# ╠═19338d41-35d2-42cd-a294-7eba3248022b
# ╠═3558803d-f591-46cd-977f-51c420b957b2
# ╠═7ece512f-3ae4-4ba4-a49d-e4ab9870f53d
# ╠═c59c8295-6c70-42ef-a48c-aa111bac8727
# ╠═fa5a50b8-aa1b-49b2-9823-eaf332268b75
# ╠═41e879d7-9d5c-438f-bb63-09b3c9b19d82
# ╠═c2c05ba4-7344-417d-a075-07dbfc3cc9a1
# ╠═1bc1ba05-7c01-4ab2-87ca-6444a634bfe7
# ╟─784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
