# =============================================================================
# roadef_checker.jl
# Julia interface to the OFFICIAL ROADEF/EURO 2010 constraint checker.
#
# This wraps `checker.jar` (the competition's authoritative validator), so the
# verdict is guaranteed to match the official checker — it *is* the checker.
# A pure-Julia reimplementation is possible but the fuel constraints (CT5/6,
# CT9–12) are subtle and would not be guaranteed to match the JAR, so for a
# validator you can trust this wrapper is the safe choice.
#
# Usage:
#   include("roadef_checker.jl"); using .RoadefChecker
#   res = check_solution("data_release_13102009/data_release_13102009/data2.txt",
#                        "sol_RLls70_data2.txt")
#   res.feasible            # true if no HARD violation (CT13bis is final-phase, soft)
#   res.constraints         # Dict "CT1"=>"OK", ..., "CT10_11"=>"VIOLATED", ...
#   res.cost                # total cost reported by the checker
#   res.hard_violations     # Vector of violated constraints excluding CT13bis
#
# Requires Java (>= 1.5) on PATH (or pass `java=` / set JAVA_HOME).
# =============================================================================
module RoadefChecker

export check_solution, is_feasible, CheckResult

# Exactly the labels checker.jar prints in constraint_check.txt (grouped pairs
# 3-4, 5_6, 10_11; CT9 is the fuel-balance equation enforced inside CT10_11).
const CT_LABELS = ["1", "2", "3-4", "8", "5_6", "7", "10_11", "12", "13",
                   "13bis", "14", "15", "16", "17", "18", "19", "20", "21"]

struct CheckResult
    feasible::Bool                      # no HARD violation (excludes CT13bis)
    constraints::Dict{String,String}    # "CT<label>" => "OK" | "VIOLATED"
    hard_violations::Vector{String}     # violated, excluding CT13bis
    violations::Vector{String}          # all violated (incl. CT13bis)
    cost::Union{Float64,Nothing}        # checker's reported total cost
    raw::String                         # the checker's REPORT text (for inspection)
end

_find_java() = begin
    j = Sys.which("java")
    j !== nothing && return j
    jh = get(ENV, "JAVA_HOME", "")
    !isempty(jh) && isfile(joinpath(jh, "bin", "java.exe")) && return joinpath(jh, "bin", "java.exe")
    !isempty(jh) && isfile(joinpath(jh, "bin", "java"))     && return joinpath(jh, "bin", "java")
    return "java"   # last resort: hope it's on PATH
end

"""
    check_solution(data_path, sol_path; jar, java, xmx="2g", timeout=600) -> CheckResult

Run the official `checker.jar` on a (instance, solution) pair and parse its
`constraint_check.txt` report. Returns a `CheckResult`.

`CT13bis` ("all outages scheduled") is a *final-phase* constraint; solutions in
the qualification phase may leave outages unscheduled, so it is reported but
**excluded** from `feasible` / `hard_violations` (matching how the project scores
feasibility). Inspect `res.violations` if you also want CT13bis.
"""
function check_solution(data_path::AbstractString, sol_path::AbstractString;
                        jar::AbstractString = joinpath("CHECKER", "CHECKER", "checker.jar"),
                        java::AbstractString = _find_java(),
                        xmx::AbstractString = "2g",
                        timeout::Real = 600)
    data_path = abspath(data_path); sol_path = abspath(sol_path); jar = abspath(jar)
    isfile(data_path) || error("instance file not found: $data_path")
    isfile(sol_path)  || error("solution file not found: $sol_path")
    isfile(jar)       || error("checker.jar not found: $jar")

    dir = mktempdir()                          # checker writes its logs into the CWD
    try
        # run the checker in `dir`; the -Xmx flag matters (data files are large)
        cmd = Cmd(`$java -Xmx$xmx -jar $jar -d $data_path -r $sol_path`; dir = dir)
        proc = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = devnull); wait = false)
        t0 = time()
        while process_running(proc)
            time() - t0 > timeout && (kill(proc); error("checker timed out after $timeout s"))
            sleep(0.1)
        end

        cc = joinpath(dir, "constraint_check.txt")
        if !isfile(cc)
            # checker only writes constraint_check.txt if the data/solution FORMAT is OK
            fmt = ""
            for f in ("solution_format_log.txt", "data_format_log.txt")
                p = joinpath(dir, f)
                isfile(p) && (fmt *= "\n--- $f ---\n" * read(p, String))
            end
            error("checker produced no constraint_check.txt (solution/data format problem?)$fmt")
        end

        txt = read(cc, String)
        report = last(split(txt, "REPORT"))    # the section after the REPORT banner

        cons = Dict{String,String}()
        for lbl in CT_LABELS
            m = match(Regex("Constraint" * replace(lbl, "-" => "\\-") * raw"\s*:([^\n]*)"), report)
            ok = m !== nothing && startswith(strip(m.captures[1]), "OK")
            cons["CT" * lbl] = ok ? "OK" : "VIOLATED"
        end
        viol = [k for (k, v) in cons if v != "OK"]
        hard = filter(!=("CT13bis"), viol)

        cm = match(r"Total cost\s*:\s*([0-9.eE+\-]+)", report)
        cost = cm === nothing ? nothing : parse(Float64, cm.captures[1])

        return CheckResult(isempty(hard), cons, hard, viol, cost, report)
    finally
        rm(dir; recursive = true, force = true)
    end
end

"Convenience: true iff the solution has no HARD constraint violation."
is_feasible(data_path, sol_path; kw...) = check_solution(data_path, sol_path; kw...).feasible

end # module

# ---------------------------------------------------------------------------
# Run as a script:  julia roadef_checker.jl <data_file> <solution_file>
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    using .RoadefChecker
    length(ARGS) >= 2 || error("usage: julia roadef_checker.jl <data_file> <solution_file>")
    r = check_solution(ARGS[1], ARGS[2])
    println("=== official checker — ", basename(ARGS[2]), " ===")
    for lbl in RoadefChecker.CT_LABELS
        println("  CT", rpad(lbl, 6), "  ", r.constraints["CT"*lbl])
    end
    println()
    println("hard violations (excl. CT13bis): ", isempty(r.hard_violations) ? "NONE  ✅" : r.hard_violations)
    println("CT13bis (final-phase, soft)    : ", r.constraints["CT13bis"])
    println("feasible                       : ", r.feasible)
    println("checker cost                   : ", r.cost)
end
