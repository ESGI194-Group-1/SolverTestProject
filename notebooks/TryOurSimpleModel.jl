### A Pluto.jl notebook ###
# v1.0.0

using Markdown
using InteractiveUtils

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

# ╔═╡ 568a1a22-217d-4c3e-a971-2ffec277930e
md"""
# Try our simple model
"""

# ╔═╡ 68f298b8-183c-4180-b9b2-b648ce0e2d67
Weeks = 1:250

# ╔═╡ 9a468887-f453-42b1-90a3-b888e12f3398
const week = 24 * 7

# ╔═╡ ecedc396-8a6d-4ae6-ab41-e295bbfe3ace
Hours = 1:(Weeks[end] * week)

# ╔═╡ cac8e501-e013-448d-b245-d2d761a03d24
@kwdef mutable struct NuclearPlant
    x = 1
    p = Float64[] # p_{i,t}
    Cfactor = 0.1 #Cost per GWh
    maxpower = 1 #GW
    burnrate = 4.0e-5 # per GWh
    replacementtime = 2 * week
    outagecount = 0
    needmaintenance = false
end

# ╔═╡ 0ccb519b-dd76-47b5-a8ee-b78c4ee8eb07
@kwdef mutable struct FossilPlant
    p = Float64[] # p_{j,t}
    Cfactor = 10 # cost per GWh
    maxpower = 5 #GW
end

# ╔═╡ e67577b1-422a-496c-8d62-973aa9c4b42b
md"""
Run reactor for an hour
"""

# ╔═╡ 05296397-6296-4b1b-aa3d-21a9abf9ae67
function run(plant::NuclearPlant, L, allowmaintenance)
    p = min(L, plant.maxpower)
    if plant.outagecount == 0
        plant.x = plant.x - plant.burnrate * p
        if plant.x < 0.3
            plant.needmaintenance=true
        end
        if plant.needmaintenance && allowmaintenance
            plant.x = 1
            plant.outagecount = plant.replacementtime
            plant.needmaintenance=false
        end
        push!(plant.p, p)
        return p, plant.Cfactor * p
    else
        push!(plant.p, 0)
        plant.outagecount -= 1
        return 0, 0
    end
end;

# ╔═╡ 0557da68-b12a-48c9-9609-23c89ca055c8
md"""
Run a fossil plant for an hour
"""

# ╔═╡ e295db18-27cd-400b-beb9-9301527eb21f
function run(plant::FossilPlant, L)
    p = min(L, plant.maxpower)
    push!(plant.p, p)
    return p, plant.Cfactor * p
end

# ╔═╡ f7fd9004-f172-4f67-a385-1f34a1338866
@kwdef struct Fleet
	nuclear=[NuclearPlant() for i=1:4]
	fossil=[FossilPlant() for i=1:1]
end

# ╔═╡ 41b261a1-16ad-4bd0-b4b0-0a936a6e862a
function run(fleet::Fleet,period, demand)
    (; nuclear, fossil)= fleet
    hour = 1
    C = 0
    pfleet = []
    for week in period
        if week==50
   #         nuclear[1].needmaintenance=true
        end

        for day in 1:7
            for h in 1:24
                L = demand(hour)
                hour = hour + 1
                np = 0
                nC = 0
                for plant in nuclear
                    # ensure only one plant can get maintenance
                    allowmaintenance=true
                    for i=1:length(nuclear)
                        if nuclear[i]!==plant && nuclear[i].outagecount>0
                            allowmaintenance=false
                        end
                    end
                    p, C = run(plant, L / length(nuclear), allowmaintenance)
                    np += p
                    nC += C
                end
                fL = L - np
                fp = 0
                fC = 0
                for plant in fossil
                    p, C = run(plant, fL / length(fossil))
                    fp += p
                    fC += C
                end
                C += nC + fC
                push!(pfleet, fp + np)
                if (fp+np) * (1.0 + 1.0e-10) < L
                    error("demand not met, $fgeneration, $ngeneration, $load")
                end
            end
        end
    end
    return fleet, pfleet, C
end

# ╔═╡ 6c12503f-6b0b-44c9-8001-2663c80788ca
clip(x, a, b) = max(a, min(x, b))

# ╔═╡ 1145d7b2-fc18-4097-9493-51fedd5e7ba6
# Source - https://stackoverflow.com/a/59565645
# Posted by Przemyslaw Szufel
# Retrieved 2026-06-02, License - CC BY-SA 4.0

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

# ╔═╡ 07df028c-9339-4a2a-a0f4-670a343ba02d
begin
    n=100
    L=zeros(length(Hours)+n-1)
    L[1]=5
    for h in 2:length(L)
        L[h]=clip(L[h-1]+0.001*randn(),3,5.5)
    end
    L=moving_average(L,n)
end

# ╔═╡ 2da0b9e0-faa3-4e89-b1eb-a8e19cc0f189
demand(hour) = L[hour] #GW

# ╔═╡ 578f9d45-02d8-473f-83c4-1c6acf89029b
plot(Hours,demand.(Hours))

# ╔═╡ 27a10bd0-8196-4e8d-9e46-9a431cf9999a
fleet, p,C = run(Fleet(), Weeks, demand)

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

# ╔═╡ d256357b-a78b-4d4e-b200-c356542def89
let
    pl=plot(legendposition=(250,1), size=(500,250), title="C=$(round(C,sigdigits=3))")
    plot!(pl, Hours/week, L, label="load")
    for i in 1:length(fleet.nuclear)
        plant=fleet.nuclear[i]
        plot!(pl, Hours/week, plant.p, label= i==1 ? "nuclear" : nothing , color=:green)
    end
    for plant in fleet.fossil
        plot!(pl, Hours/week, plant.p, label="fossil", color=:red)
    end
pl
end |> (x)->floataside(x, top=20, width=500)

# ╔═╡ Cell order:
# ╠═bdee4d8e-e655-497d-9acb-3d23f1a5b68c
# ╟─568a1a22-217d-4c3e-a971-2ffec277930e
# ╠═68f298b8-183c-4180-b9b2-b648ce0e2d67
# ╠═9a468887-f453-42b1-90a3-b888e12f3398
# ╠═ecedc396-8a6d-4ae6-ab41-e295bbfe3ace
# ╠═cac8e501-e013-448d-b245-d2d761a03d24
# ╠═0ccb519b-dd76-47b5-a8ee-b78c4ee8eb07
# ╟─e67577b1-422a-496c-8d62-973aa9c4b42b
# ╠═05296397-6296-4b1b-aa3d-21a9abf9ae67
# ╟─0557da68-b12a-48c9-9609-23c89ca055c8
# ╠═e295db18-27cd-400b-beb9-9301527eb21f
# ╠═f7fd9004-f172-4f67-a385-1f34a1338866
# ╠═41b261a1-16ad-4bd0-b4b0-0a936a6e862a
# ╠═6c12503f-6b0b-44c9-8001-2663c80788ca
# ╠═1145d7b2-fc18-4097-9493-51fedd5e7ba6
# ╠═07df028c-9339-4a2a-a0f4-670a343ba02d
# ╠═578f9d45-02d8-473f-83c4-1c6acf89029b
# ╠═2da0b9e0-faa3-4e89-b1eb-a8e19cc0f189
# ╠═27a10bd0-8196-4e8d-9e46-9a431cf9999a
# ╠═d256357b-a78b-4d4e-b200-c356542def89
# ╠═e209b2a5-36b8-4464-8527-d95fd371ed7c
