using Gridap, HyperFEM, HyperShape, GridapSolvers, DrWatson
using GridapSolvers.NonlinearSolvers
using GridapSolvers.LinearSolvers
using Gridap.FESpaces
using WriteVTK
using HyperShape:evaluate!

function run()
pname = "CantileverNeumann_phasefield"
simdir = datadir("sims", pname)
setupfolder(simdir)

xmax = 3.0
ymax = 1.0
domain = (0, xmax, 0, ymax)
N = 80
h = 1 / N
partition = (3N, N)
geomodel = CartesianDiscreteModel(domain, partition)

# Define new boundaries
labels = get_face_labeling(geomodel)
add_tag_from_tags!(labels, "diri_0", [1, 3, 7])
add_tag_from_tags!(labels, "Force", [2, 4, 8])

physmodel_s = MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=0.0, ρ=1.0)
physmodel_v = LinearElasticity2D(λ=3.0 * 1e-3, μ=1.0 * 1e-3, ρ=1.0)
physmodel = EnergyInterpolationScheme(physmodel_s, physmodel_v; p=3.0)

_, ∂Ψ, ∂2Ψ, _, D∂Ψ_Dρ, _ = physmodel()

# Setup integration
order = 1
degree = 4 * order
Ω = Triangulation(geomodel)
dΩ = Measure(Ω, degree)
Γ = BoundaryTriangulation(geomodel, tags="Force")
dΓ = Measure(Γ, degree)

# Dirichlet conditions 
dir_u_tags = ["diri_0"]
dir_u_values = [[0.0, 0.0]]
D_bc = DirichletBC(dir_u_tags, dir_u_values)

#  FE elements 
reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
refρ = ReferenceFE(lagrangian, Float64, 1)

V = TestFESpace(geomodel, reffeu, D_bc)
Vρ = TestFESpace(Ω, refρ, NothingBC())
U = TrialFESpace(V, D_bc, 1.0)

# ************************************************************************************************
#                                 Density initialization
# ************************************************************************************************
Number_holes = [7, 3]
radius_holes = 0.1

x = range(0.15, stop=xmax - 0.1, length=Number_holes[1])  # Posiciones en x
y = range(0.15, stop=ymax - 0.15, length=Number_holes[2])  # Posiciones en y

centros = [collect(c) for c in reshape(collect(Iterators.product(x, y)), Number_holes[1], Number_holes[2])]  # Generar matriz de vectores

centros = centros[:]
shapes = Vector{Any}(map((x) -> Ellipsoid(x, [radius_holes, radius_holes]), centros))
box1 = Box([0.0, 0.0], [2.5 * radius_holes, ymax])
box2 = Box([xmax - radius_holes, 0.0], [radius_holes + 0.1, ymax])

push!(shapes, box1)
push!(shapes, box2)

ρinit(y) = 1.0 - sum(map((x) -> x(y), shapes[1:end-2])) + sum(map((x) -> x(y), shapes[end-2:end]))

ρh = interpolate_everywhere(ρinit, Vρ) # variable en L2
ρh_ = get_free_dof_values(ρh)
ρh_ .= map((x) -> max(0.0, min(x, 1.0)), ρh_)

fileI_seed = datadir("sims", pname) * "/seed" * ".vtu"
writevtk(Ω, fileI_seed, cellfields=["ρh" => ρh])


# ************************************************************************************************
#                                 Boundary FILTER
# ************************************************************************************************

ε = 2h
a_filter(u, v) = ∫(ε^2 * (∇(v) ⋅ ∇(u)) + v * u) * dΩ
FILTER = StaticLinearModel(a_filter, Vρ, Vρ, D_bc)


# ************************************************************************************************
#                                 Forward Problem
# ************************************************************************************************

km = Kinematics(Mechano, Solid)
F, _, _ = get_Kinematics(km)
ΔDirac(ϵ, x₁, x₂) = (x) -> (x[1] ≥ x₁ - ϵ && x[1] ≤ x₁ + ϵ && x[2] ≥ x₂ - ϵ && x[2] ≤ x₂ + ϵ) ? 1.0 : 0.0
g(Λ) = (x) -> ΔDirac(0.1, xmax, ymax / 2)(x) * VectorValue(0.0, -Λ * 5e-3)
res(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ ∘ (ρh, F ∘ (∇(u)'))))dΩ - ∫(g(Λ) ⋅ v)dΓ
jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂2Ψ ∘ (ρh, F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩ

ls = LUSolver()
nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-8, rtol=1.e-6, verbose=false)
FORWARD = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls_)

uh = get_state(FORWARD)
 
# ************************************************************************************************
#                                 Adjoint Problem
# ************************************************************************************************

a(p, v) = jac(1.0)(uh, p, v)
l(v) = ∫(g(1.0) ⋅ v)dΓ
ADJOINT = StaticLinearModel(l, a, V, V, D_bc)

ph = get_state(ADJOINT)


# ************************************************************************************************
#                                 Optimisation functionals
# ************************************************************************************************

J_(u, ϕ) = ∫(g(1.0) ⋅ u)dΓ
DJ_(u, p, ϕ) = (v) -> (-1.0) * ∫((((D∂Ψ_Dρ ∘ (ϕ, F ∘ (∇(u)'))) ⊙ (∇(p)')) * ρh * (1.0 - ρh)) * v)dΩ
Compliance = FEFunctional(J_, DJ_, uh, ph, ρh)

Vol_(u, ϕ) = ∫(ϕ)dΩ
DVol_(u, p, ϕ) = (v) -> ∫(ρh * (1.0 - ρh) * v)dΩ
Volume = FEFunctional(Vol_, DVol_, nothing, nothing, ρh)

# ************************************************************************************************
#                                 Optimization loop
# ************************************************************************************************

tol = 1e-3
λ = 10
t = 0.24
V0 = ∑(∫(1.0)dΩ)
filePath = datadir("sims", pname)
pvd_topo = paraview_collection(filePath, append=false)

PDES = [FORWARD, ADJOINT]

options = Dict(
  FORWARD => (stepping=(nsteps=1, maxbisec=5),),
  ADJOINT => NamedTuple(),
)

ρv = get_free_dof_values(ρh)

  for i_top in 1:200

    # SOLVE PDEs
    map(x -> solve!(x; options[x]...), PDES)

    # Evaluate Functionals
    J, dJ = evaluate!(Compliance)
    Vol, dVol = evaluate!(Volume)

    if i_top == 1
      J, dJ = adimensionalize!(Compliance, J)
      Vol, dVol = adimensionalize!(Volume, V0)
      options[FORWARD] = (stepping=(nsteps=1, maxbisec=5),)
    end

    FILTER(dJ)
    FILTER(dVol)

    Jt = J + λ * Vol
    DJt = dJ + λ * dVol

    ρv .-= t * DJt
    ρv .= map((x) -> max(0.0, min(x, 1.0)), ρv)
    @show J, Vol, Jt, norm(DJt), minimum(ρv), maximum(ρv)
    DJht = FEFunction(Vρ, DJt)
    if norm(DJt) < tol
      break
    else
      fileI = filePath * "/iter_$i_top" * ".vtu"
      pvd_topo[i_top] = createvtk(Ω, fileI, cellfields=["uh" => uh, "ρh" => ρh, "DJh" => DJht])
    end
  end
  WriteVTK.vtk_save(pvd_topo)

end

run()

