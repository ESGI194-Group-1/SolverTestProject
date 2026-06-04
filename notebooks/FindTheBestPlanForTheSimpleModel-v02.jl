### A Pluto.jl notebook ###
# v1.0.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ bdee4d8e-e655-497d-9acb-3d23f1a5b68c
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    import PlotlyBase, PlotlyKaleido, PlotlyJS
    using Plots
    using SparseArrays
    using PlutoUI, HypertextLiteral, UUIDs
    plotly()
    Plots.default(
        linewidth = 2,
        gridwidth = 2,
        framestyle = :box,
        size = (475, 250)
    )

end

# ╔═╡ cc2cc00d-a295-427f-a521-9826158e79a9
TableOfContents()

# ╔═╡ 568a1a22-217d-4c3e-a971-2ffec277930e
md"""
# Find the best plan for the simple model
"""

# ╔═╡ d0d3cd96-48cb-49a3-b207-cae03ccfb1db
md"""
### Time intervals
"""

# ╔═╡ 68f298b8-183c-4180-b9b2-b648ce0e2d67
Weeks = 1:250

# ╔═╡ 9a468887-f453-42b1-90a3-b888e12f3398
const week = 7

# ╔═╡ ecedc396-8a6d-4ae6-ab41-e295bbfe3ace
Days = 1:(Weeks[end] * week)

# ╔═╡ f0a4f42f-dd63-4c14-8050-c428218fbc24
md"""
### Power plants
"""

# ╔═╡ cac8e501-e013-448d-b245-d2d761a03d24
@kwdef mutable struct NuclearPlant
    P = 1         # maximum power GW
    Xᵐᵃˣ = 1      # maximum inventory
    x = Xᵐᵃˣ      # nuclear inventory x
    p = Float64[] # power history p_{i,t}
    C = 0.1       # cost C_{i,t} per GWd
    c = 2.0e-4    # inventory drawdown per GWd
end; 

# ╔═╡ e67577b1-422a-496c-8d62-973aa9c4b42b
md"""
Run reactor for a day with assigned load and maintenance plan
"""

# ╔═╡ 05296397-6296-4b1b-aa3d-21a9abf9ae67
function run(
        plant::NuclearPlant,
        L, # Load assigned to plant
        d  # Maintenance if d==1
    )
    if plant.x > 0
        if d == 1
            plant.x = plant.Xᵐᵃˣ # refuel
            p = 0
        else
            p = min(L, plant.P)
            plant.x = plant.x - plant.c * p
        end
    else
        p = 0
    end
    push!(plant.p, p)
    return p, plant.C * p
end;

# ╔═╡ 0ccb519b-dd76-47b5-a8ee-b78c4ee8eb07
@kwdef mutable struct FossilPlant
    P = 1         # maximum power GW
    p = Float64[] # power history p_{j,t}
    C = 10        # cost C_{j,t} per GWd
end

# ╔═╡ 0557da68-b12a-48c9-9609-23c89ca055c8
md"""
Run a fossil plant for a day
"""

# ╔═╡ e295db18-27cd-400b-beb9-9301527eb21f
function run(
        plant::FossilPlant,
        L # Load assigned to plant
    )
    p = min(L, plant.P)
    push!(plant.p, p)
    return p, plant.C * p
end;

# ╔═╡ bfe2f001-2697-4570-872a-f0eca332aa6a
md"""
### Fleet
"""

# ╔═╡ f7fd9004-f172-4f67-a385-1f34a1338866
@kwdef struct Fleet
    nuclear = [NuclearPlant() for i in 1:4]
    fossil = [FossilPlant() for i in 1:5]
end;

# ╔═╡ 84239ceb-28ca-452a-8513-ceaca396defc
md"""
Unit commitment

$p^N_{i,t} = \min\left(\frac{L_t}{|N|},\overline P_i(1-d_{i,t})\right)$

$p^F_{j,t} = \min\left( \frac{L_t - \sum_{i\in N}p^N_{i,t}}{|F|},  \overline P_j\right)$
"""

# ╔═╡ 3f211ff7-0cc1-4433-b6c8-9d3865dffaf9
md"""
Run the fleet for period of time (range of weeks), with given load L and given maintenance plan d. 
- Returns the fleet, the power history of the fleet and the cost
```math    
\sum_t \sum_{j \in F}C_{j,t}\, p^{F}_{j,t}(d,L_t)
```
- Generates and records plant output $p^F$ and $p^N$ in the 'p' vectors of the plants
__TODO__: formalize this as a linear operator ?
"""

# ╔═╡ 41b261a1-16ad-4bd0-b4b0-0a936a6e862a
function run(
        fleet::Fleet,
        period, # range of weeks
        L,      # daily loadime
        d       # maintenance schedule
    )
    day = 0
    Cfleet = 0.0
    pfleet = []
    for week in period
        for weekday in 1:7
            day = day + 1
            @views p, C = unitcommitment(fleet, L[day], d[:, week])
            Cfleet += C
            push!(pfleet, p)
            if p * (1.0 + 1.0e-10) < L[day]
                throw(ErrorException("demand not met, $p, $L[day]"))
                Cfleet += 1000
            end
        end
    end
    return fleet, pfleet, Cfleet
end;

# ╔═╡ ac6cae6c-5423-46c6-80c1-68425cef35bd
function unitcommitment(fleet::Fleet, L, d)
    (; nuclear, fossil) = fleet

    # First commit to nuclear
    np = 0.0
    nC = 0.0
    for inuclear in 1:length(nuclear)
        p, C = run(nuclear[inuclear], L / length(nuclear), d[inuclear])
        np += p
        nC += C
    end

    # commit the rest to fossil
    fL = L - np
    fp = 0.0
    fC = 0.0
    for plant in fossil
        p, C = run(plant, fL / length(fossil))
        fp += p
        fC += C
    end

    return fp + np, fC + nC
end;

# ╔═╡ 4197dda8-c4a7-4f30-ad90-d5c079fd19d9
md"""
### Synthetic load generation
"""

# ╔═╡ 6c12503f-6b0b-44c9-8001-2663c80788ca
clip(x, a, b) = max(a, min(x, b));

# ╔═╡ 1145d7b2-fc18-4097-9493-51fedd5e7ba6
# Source - https://stackoverflow.com/a/59565645
# Posted by Przemyslaw Szufel
# Retrieved 2026-06-02, License - CC BY-SA 4.0
moving_average(vs, n) = [sum(@view vs[i:(i + n - 1)]) / n for i in 1:(length(vs) - (n - 1))];

# ╔═╡ 578f9d45-02d8-473f-83c4-1c6acf89029b
#plot(Days,demand.(Days))

# ╔═╡ 07c36b04-c0cc-4ba3-bc27-ffa68d433aa2
md"""
### Maintenance plan

Try to create a maintenance plan (``d_{i,t}``) with constraints
"""

# ╔═╡ 6ed68fbb-97ab-45e5-82fd-3514601a353d
md"""
Check feasibility of the plan
"""

# ╔═╡ 0747805d-c375-4390-8887-43caccad319b
function isfeasible(d; maxparallel, nmaint, duration)
    # Check maxparallel constraint
    @assert all(s -> s ≤ maxparallel, sum(d, dims = 1))
    # Check if all plants get nmaint maintenances
    @assert all(s -> s == nmaint * duration, sum(d, dims = 2))
    return true
end;

# ╔═╡ d6be922c-30fb-419d-bf48-c87867353700
function maintenanceplan(
        fleet, weeks;
        nmaint = 3,         # number of maintenances per plant
        duration = 2,       # duration of maintenance
        maxparallel = 1,    # number of plants which can have maint at same time
        weekstart = 10,     # first week of maintenance for plant 1
        rand_interval = 15, # randomization of maintenance interval
        rand_duration = 5  # randomization of maintenance interval between plants
    )
    nplants = length(fleet.nuclear)
    nweeks = length(weeks)
    d = zeros(Int, nplants, nweeks)
    w0 = weekstart
    interval = Int(ceil(nweeks / nmaint))
    for iplant in 1:nplants
        wx = w0
        for imaint in 1:nmaint
            # Ensure maxparallel constraint
            while !all(
                    s -> s ≤ maxparallel - 1,
                    sum(d[1:(iplant - 1), wx:(wx + duration - 1)], dims = 1)
                )
                # if other plant(s) are under maintenace, shift possible schedule
                wx = wx + 1
            end
            # Reserve maintenance for this plant iplant
            for idur in 1:duration
                d[iplant, wx] = 1
                wx = wx + 1
            end
            # No maintenance during this interval; randomized
            wx += rand((interval - rand_interval):(interval + rand_interval))
        end
        # shift first week for next plant
        w0 = w0 + duration + rand(1:rand_duration)
    end
    @assert isfeasible(d; nmaint, maxparallel, duration)
    return d
end;

# ╔═╡ 37176eb8-096b-4f87-bcb7-d2b97de84923
md"""
Create a fleet with a maintenance plan
"""

# ╔═╡ 5181596d-2f5e-4265-92d6-7a5d29f93dc8
function makefleet(weeks; kwargs...)
    fleet = Fleet(; kwargs...)
    plan = maintenanceplan(fleet, weeks)
    return fleet, plan
end;

# ╔═╡ 8154d08a-a965-4919-8c0d-d3bede000bda
md"""
### Brute force optimization

Run 500 instances and identify the plan with the least cost
"""

# ╔═╡ 52b2aed3-41c9-4570-ac87-1ee2f57de5d9
function findbestplan(L; nrun = 500)
    Cmin = 1.0e30
    bestplan = nothing
    bestfleet = nothing
    for irun in 1:nrun
        fleet, d = makefleet(Weeks)
        fleet, p, C = run(fleet, Weeks, L, d)
        if C < Cmin
            bestplan = d
            bestfleet = fleet
            Cmin = C
        end
    end
    return bestfleet, Cmin
end;

# ╔═╡ 719a32ee-81c4-471f-b2d5-5fdc42dbd221
md"""
### Result plot
"""

# ╔═╡ a495c8ab-5286-4cd3-b84a-c75efadf7d36
md"""
__Appendix__:
Package imports
"""

# ╔═╡ fb102bed-b6ae-4689-9fa7-e8e480a2b92e
#myplot(; size=(750,300), save="fig_load.png")

# ╔═╡ e209b2a5-36b8-4464-8527-d95fd371ed7c
begin
    function floataside(text::Markdown.MD; top = 1, width = 500)
        uuid = uuid1()
        return @htl(
            """
            	<style>
                 @media (min-width: calc(700px + 30px + 300px)) 
                  {
            		aside.plutoui-aside-wrapper-$(uuid) 
                      {
            	        color: var(--pluto-output-color);
            	        position:fixed;
            	        right: 0.1rem;
            	        top: $(top)px;
            	        width: $(width)px;
            	        padding: 5px;
            	        border: 2px solid rgba(0, 0, 0, 0.15);
            	        border-radius: 5px;
                    	overflow: auto;
                    	z-index: 40;
                    	background-color: var(--main-bg-color);
                      }
            	}
            	</style>
            	<aside class="plutoui-aside-wrapper-$(uuid)">
            	<div>$(text)</div>
            	</aside>
            	"""
        )
    end
    floataside(stuff; kwargs...) = floataside(md"""$(stuff)"""; kwargs...)
end;

# ╔═╡ 04fd9413-2160-45f8-9b76-643e460ae303
floataside(
    md"""
    $(@bind trigger_update_load PlutoUI.Button("Update Load")) 
    $(@bind trigger_rerun PlutoUI.Button("Rerun Optimization")) 
    """; top = 320
)

# ╔═╡ 07df028c-9339-4a2a-a0f4-670a343ba02d
begin
    trigger_update_load # re-run this is "Update Load" is pressed
    navg = 10           # rolling average window
    Linitial = 4        # initial load
    LMinMax = (3, 5)    # min/max load
    ψvar = 0.03         # pseudo variance (for randomness)

    L = zeros(length(Days) + navg - 1)
    L[1] = Linitial
    for h in 2:length(L)
        L[h] = clip(L[h - 1] + ψvar * randn(), LMinMax...)
    end
    L = moving_average(L, navg)
end

# ╔═╡ 58241d1e-c4cc-4d9c-8edb-874653b89abc
begin
    trigger_rerun # Re-run this cell if "Rerun Optimization" is pressed
    fleet, C = findbestplan(L)
end

# ╔═╡ 43d06523-9d98-4f34-a851-b29e5eac5b12
function myplot(; size=(500,200), save=nothing)
 pl = plot(;
           legendposition = (250, 1), 
         #  legendposition = :right, 
           size, title = "C=$(C)", 
           xlabel="t/week", ylabel="p/GW")
    plot!(pl, Days / week, L, label = "Load L")
    plot!(pl, Days / week, sum([plant.p for plant in fleet.nuclear]), label= "Nuclear", color = :green)
    plot!(pl, Days / week, sum([plant.p for plant in fleet.fossil]), label="Fossil", color = :red)
    if !isnothing(save)
        Plots.savefig(pl,save)
    end
    pl
end

# ╔═╡ d256357b-a78b-4d4e-b200-c356542def89
floataside(myplot(), top = 375, width = 500)

# ╔═╡ Cell order:
# ╟─cc2cc00d-a295-427f-a521-9826158e79a9
# ╟─568a1a22-217d-4c3e-a971-2ffec277930e
# ╟─d0d3cd96-48cb-49a3-b207-cae03ccfb1db
# ╠═68f298b8-183c-4180-b9b2-b648ce0e2d67
# ╠═9a468887-f453-42b1-90a3-b888e12f3398
# ╠═ecedc396-8a6d-4ae6-ab41-e295bbfe3ace
# ╟─f0a4f42f-dd63-4c14-8050-c428218fbc24
# ╠═cac8e501-e013-448d-b245-d2d761a03d24
# ╟─e67577b1-422a-496c-8d62-973aa9c4b42b
# ╠═05296397-6296-4b1b-aa3d-21a9abf9ae67
# ╠═0ccb519b-dd76-47b5-a8ee-b78c4ee8eb07
# ╟─0557da68-b12a-48c9-9609-23c89ca055c8
# ╠═e295db18-27cd-400b-beb9-9301527eb21f
# ╟─bfe2f001-2697-4570-872a-f0eca332aa6a
# ╠═f7fd9004-f172-4f67-a385-1f34a1338866
# ╟─84239ceb-28ca-452a-8513-ceaca396defc
# ╠═ac6cae6c-5423-46c6-80c1-68425cef35bd
# ╟─3f211ff7-0cc1-4433-b6c8-9d3865dffaf9
# ╠═41b261a1-16ad-4bd0-b4b0-0a936a6e862a
# ╟─4197dda8-c4a7-4f30-ad90-d5c079fd19d9
# ╠═6c12503f-6b0b-44c9-8001-2663c80788ca
# ╠═1145d7b2-fc18-4097-9493-51fedd5e7ba6
# ╠═07df028c-9339-4a2a-a0f4-670a343ba02d
# ╟─578f9d45-02d8-473f-83c4-1c6acf89029b
# ╟─07c36b04-c0cc-4ba3-bc27-ffa68d433aa2
# ╠═d6be922c-30fb-419d-bf48-c87867353700
# ╟─6ed68fbb-97ab-45e5-82fd-3514601a353d
# ╠═0747805d-c375-4390-8887-43caccad319b
# ╟─37176eb8-096b-4f87-bcb7-d2b97de84923
# ╠═5181596d-2f5e-4265-92d6-7a5d29f93dc8
# ╟─8154d08a-a965-4919-8c0d-d3bede000bda
# ╠═52b2aed3-41c9-4570-ac87-1ee2f57de5d9
# ╠═58241d1e-c4cc-4d9c-8edb-874653b89abc
# ╟─719a32ee-81c4-471f-b2d5-5fdc42dbd221
# ╠═04fd9413-2160-45f8-9b76-643e460ae303
# ╠═43d06523-9d98-4f34-a851-b29e5eac5b12
# ╠═d256357b-a78b-4d4e-b200-c356542def89
# ╟─a495c8ab-5286-4cd3-b84a-c75efadf7d36
# ╠═bdee4d8e-e655-497d-9acb-3d23f1a5b68c
# ╠═fb102bed-b6ae-4689-9fa7-e8e480a2b92e
# ╟─e209b2a5-36b8-4464-8527-d95fd371ed7c
