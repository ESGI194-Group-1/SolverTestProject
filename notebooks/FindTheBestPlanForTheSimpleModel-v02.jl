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
    using PlotlyBase, PlotlyKaleido
    using Plots
    using PlutoUI, HypertextLiteral, UUIDs
    plotly()
    Plots.default(linewidth=2,
                  gridwidth=2,
                  framestyle=:box,
                  size=(475,250))

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
    x = 1
    p = Float64[] # p_{i,t}
    Cfactor = 0.1 #Cost per GWh
    maxpower = 1 #GW
    burnrate = 1.0e-5 # per GWh
end

# ╔═╡ 0ccb519b-dd76-47b5-a8ee-b78c4ee8eb07
@kwdef mutable struct FossilPlant
    p = Float64[] # p_{j,t}
    Cfactor = 10 # cost per GWh
    maxpower = 5 #GW
end

# ╔═╡ e67577b1-422a-496c-8d62-973aa9c4b42b
md"""
Run reactor for a day
"""

# ╔═╡ 05296397-6296-4b1b-aa3d-21a9abf9ae67
function run(plant::NuclearPlant, L)
    if plant.x>0
        p = min(L, plant.maxpower)
        plant.x = plant.x - plant.burnrate * p * 24
    else
        p=0 # we will just penalize missing demand later
    end
    push!(plant.p, p)
    return p, plant.Cfactor * p *24.0
end;

# ╔═╡ 0557da68-b12a-48c9-9609-23c89ca055c8
md"""
Run a fossil plant for an hour
"""

# ╔═╡ e295db18-27cd-400b-beb9-9301527eb21f
function run(plant::FossilPlant, L)
    p = min(L, plant.maxpower)
    push!(plant.p, p)
    return p, plant.Cfactor * p *24.0
end

# ╔═╡ bfe2f001-2697-4570-872a-f0eca332aa6a
md"""
### Fleet
"""

# ╔═╡ f7fd9004-f172-4f67-a385-1f34a1338866
@kwdef struct Fleet
	nuclear=[NuclearPlant() for i=1:4]
	fossil=[FossilPlant() for i=1:1]
end

# ╔═╡ 3f211ff7-0cc1-4433-b6c8-9d3865dffaf9
md"""
Run the fleet for period of time (range of weeks), with given demand and given maintenance plan. Returns the fleet, the power history of the fleet and the cost
"""

# ╔═╡ 41b261a1-16ad-4bd0-b4b0-0a936a6e862a
function run(fleet::Fleet,period, demand, plan)
    (; nuclear, fossil)= fleet
    day = 1
    Cfleet = 0.0
    pfleet = []
    for week in period
        for weekday in 1:7
                L = demand(day)
                day = day + 1
                np = 0.0
                nC = 0.0
                for inuclear=1:length(nuclear)
                    plant=nuclear[inuclear]
                    if plan[inuclear,week] == 1
                        plant.x=1
                        push!(plant.p,0)
                    else
                        p, C = run(plant, L / length(nuclear))
                      np += p
                        nC += C
                    end 
                end
                fL = L - np
                fp = 0.0
                fC = 0.0
                for plant in fossil
                    p, C = run(plant, fL / length(fossil))
                    fp += p
                    fC += C
                end
                Cfleet += nC+fC
                push!(pfleet, fp + np)
                if (fp+np) * (1.0 + 1.0e-10) < L
                    @warn("demand not met, $fp, $np, $L")
                    Cfleet+=1000
                end
            end
    end
    return fleet, pfleet, Cfleet
end;

# ╔═╡ 4197dda8-c4a7-4f30-ad90-d5c079fd19d9
md"""
### Tooling for synthetic demand generation
"""

# ╔═╡ 6c12503f-6b0b-44c9-8001-2663c80788ca
clip(x, a, b) = max(a, min(x, b))

# ╔═╡ 1145d7b2-fc18-4097-9493-51fedd5e7ba6
# Source - https://stackoverflow.com/a/59565645
# Posted by Przemyslaw Szufel
# Retrieved 2026-06-02, License - CC BY-SA 4.0

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

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
function isfeasible(weeklyplan; maxparallel, nmaint, duration)
	# Check maxparallel constraint
	@assert all(s->s≤maxparallel, sum(weeklyplan,dims=1))
	# Check if all plants get nmaint maintenances
	@assert all(s->s==nmaint*duration, sum(weeklyplan,dims=2))
return true
end;

# ╔═╡ d6be922c-30fb-419d-bf48-c87867353700
function maintenanceplan(fleet, weeks;
						 nmaint=3, # number of maintenances per plant
						duration=2, # duration of maintenance
						 maxparallel=1, # number of plants which can have maint at same time
						 weekstart = 10 # first week of maintenance for plant 1
					)
	(;nuclear)=fleet
	nweeks=length(weeks)
	nplants=length(nuclear)
	weeklyplan=zeros(Int,nplants,nweeks)
	w0=weekstart
	interval=Int(ceil(nweeks/nmaint))
	for iplant=1:nplants
		wx=w0
		for imaint=1:nmaint
			# look for maintenance slot for available for duration
			while !all( 
				s->s≤maxparallel-1,
				sum(weeklyplan[1:(iplant-1),wx:wx+duration-1],dims=1)
				)
				wx=wx+1
			end
			# Reserve time for this plant iplant
			for idur=1:duration
				weeklyplan[iplant,wx]=1
				wx=wx+1
			end
			# No maintenance during this interval; randomized
			wx+=rand(interval-5:interval+5)
		end
		w0=w0+duration+rand(1:10)
	end
	@assert isfeasible(weeklyplan; nmaint, maxparallel, duration)
	return weeklyplan
end;

# ╔═╡ 37176eb8-096b-4f87-bcb7-d2b97de84923
md"""
Create a fleet with a maintenance plan
"""

# ╔═╡ 5181596d-2f5e-4265-92d6-7a5d29f93dc8
function makefleet(weeks; kwargs...)
	fleet=Fleet(;kwargs...)
	plan=maintenanceplan(fleet, weeks)
	return fleet, plan
end;

# ╔═╡ 8154d08a-a965-4919-8c0d-d3bede000bda
md"""
### Monte Carlo like optimization
"""

# ╔═╡ 52b2aed3-41c9-4570-ac87-1ee2f57de5d9
function findbestplan(demand; nrun=500)
	Cmin=1.0e30
	bestplan=nothing
	bestfleet=nothing
	for irun=1:nrun
		fleet,plan=makefleet(Weeks)
	 	 fleet,p,C=run(fleet, Weeks, demand, plan)
		if C<Cmin
			bestplan=plan
			bestfleet=fleet
			Cmin=C
		end
	end
	return bestfleet,Cmin
end;

# ╔═╡ 719a32ee-81c4-471f-b2d5-5fdc42dbd221
md"""
### Result plot
"""

# ╔═╡ e209b2a5-36b8-4464-8527-d95fd371ed7c
begin
    function floataside(text::Markdown.MD; top = 1, width=500)
        uuid = uuid1()
        @htl("""
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
       	""")
    end
    floataside(stuff; kwargs...) = floataside(md"""$(stuff)"""; kwargs...)
end;

# ╔═╡ 04fd9413-2160-45f8-9b76-643e460ae303
floataside(md"""
$(@bind updd PlutoUI.Button("Update Demand")) $(@bind rerun PlutoUI.Button("Rerun Solver")) 
"""; top=300)

# ╔═╡ 07df028c-9339-4a2a-a0f4-670a343ba02d
begin
    updd
    navg=10 # rolling average window
    Linitial = 5 # initial load
    L=zeros(length(Days)+navg-1)
    L[1]=Linitial
    for h in 2:length(L)
        L[h]=clip(L[h-1]+0.02*randn(),3,5.5)
    end
    L=moving_average(L,navg)
end

# ╔═╡ 2da0b9e0-faa3-4e89-b1eb-a8e19cc0f189
demand(day) = L[day] #GW

# ╔═╡ 58241d1e-c4cc-4d9c-8edb-874653b89abc
begin
	rerun
	fleet, C=findbestplan(demand)
end

# ╔═╡ d256357b-a78b-4d4e-b200-c356542def89
let
    pl=plot(legendposition=(250,1), size=(500,250), title="C=$(C)")
    plot!(pl, Days/week, L, label="load")
    for i in 1:length(fleet.nuclear)
        plant=fleet.nuclear[i]
        plot!(pl, Days/week, plant.p, label= i==1 ? "nuclear" : nothing , color=:green)
    end
    for plant in fleet.fossil
        plot!(pl, Days/week, plant.p, label="fossil", color=:red)
    end
pl
end |> (x)->floataside(x, top=375, width=500)

# ╔═╡ Cell order:
# ╠═bdee4d8e-e655-497d-9acb-3d23f1a5b68c
# ╟─cc2cc00d-a295-427f-a521-9826158e79a9
# ╟─568a1a22-217d-4c3e-a971-2ffec277930e
# ╟─d0d3cd96-48cb-49a3-b207-cae03ccfb1db
# ╠═68f298b8-183c-4180-b9b2-b648ce0e2d67
# ╠═9a468887-f453-42b1-90a3-b888e12f3398
# ╠═ecedc396-8a6d-4ae6-ab41-e295bbfe3ace
# ╟─f0a4f42f-dd63-4c14-8050-c428218fbc24
# ╠═cac8e501-e013-448d-b245-d2d761a03d24
# ╠═0ccb519b-dd76-47b5-a8ee-b78c4ee8eb07
# ╟─e67577b1-422a-496c-8d62-973aa9c4b42b
# ╠═05296397-6296-4b1b-aa3d-21a9abf9ae67
# ╟─0557da68-b12a-48c9-9609-23c89ca055c8
# ╠═e295db18-27cd-400b-beb9-9301527eb21f
# ╟─bfe2f001-2697-4570-872a-f0eca332aa6a
# ╠═f7fd9004-f172-4f67-a385-1f34a1338866
# ╟─3f211ff7-0cc1-4433-b6c8-9d3865dffaf9
# ╠═41b261a1-16ad-4bd0-b4b0-0a936a6e862a
# ╟─4197dda8-c4a7-4f30-ad90-d5c079fd19d9
# ╠═6c12503f-6b0b-44c9-8001-2663c80788ca
# ╠═1145d7b2-fc18-4097-9493-51fedd5e7ba6
# ╠═04fd9413-2160-45f8-9b76-643e460ae303
# ╠═07df028c-9339-4a2a-a0f4-670a343ba02d
# ╠═578f9d45-02d8-473f-83c4-1c6acf89029b
# ╠═2da0b9e0-faa3-4e89-b1eb-a8e19cc0f189
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
# ╠═d256357b-a78b-4d4e-b200-c356542def89
# ╟─e209b2a5-36b8-4464-8527-d95fd371ed7c
