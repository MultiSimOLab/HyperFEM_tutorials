using Gridap, GridapGmsh, HyperFEM, HyperShape, GridapSolvers, DrWatson
using TimerOutputs
using GridapSolvers.NonlinearSolvers
using HyperShape:evaluate!
using WriteVTK

pname = "CantileverNeuman_density"
simdir = datadir("sims", pname)
setupfolder(simdir)

xmax=3.0
ymax=1.0
domain = (0, xmax, 0, ymax)
N = 30
h = 1 / N
partition = (3N, N)
geomodel = CartesianDiscreteModel(domain, partition)

# Define new boundaries
labels = get_face_labeling(geomodel)
add_tag_from_tags!(labels, "diri_0", [1, 3, 7])
add_tag_from_tags!(labels, "Force", [2, 4, 8])


physmodel_s = MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=0.0, ρ=1.0)
physmodel_v = MooneyRivlin2D(λ=3.0 * 1e-6, μ1=1.0 * 1e-6, μ2=0.0, ρ=1.0)
physmodel = EnergyInterpolationScheme(physmodel_s, physmodel_v; p=3.0)

_, ∂Ψ, ∂2Ψ, _, D∂Ψ_Dρ, _ = physmodel()

# Setup integration
order = 1
degree = 2 * order
Ω = Triangulation(geomodel)
dΩ = Measure(Ω, degree)
Γ = BoundaryTriangulation(geomodel, tags="Force")
dΓ = Measure(Γ, degree)

# Dirichlet conditions 
evolu(Λ) = 1.0
dir_u_tags = ["diri_0"]
dir_u_values = [[0.0, 0.0]]
dir_u_timesteps = [evolu]
D_bc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

#  FE elements 
reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
refρL2 = ReferenceFE(lagrangian, Float64, 0)
refρ = ReferenceFE(lagrangian, Float64, 1)

V = TestFESpace(geomodel, reffeu, D_bc, conformity=:H1)
VρL2 = TestFESpace(Ω, refρL2, NothingBC(), conformity=:L2)
Vρ = TestFESpace(Ω, refρ, NothingBC(), conformity=:H1)
U = TrialFESpace(V, D_bc)

# ************************************************************************************************
#                                 Density initialization
# ************************************************************************************************
ρinit_ = Ellipsoid([0.5, 0.5], [0.2, 0.2])
ρinit(x) = 1.0 + 0.0 * ρinit_(x)
ρh = interpolate_everywhere(ρinit, VρL2) # variable en L2

# ************************************************************************************************
#                                 Filter density
# ************************************************************************************************

ε = 3h
a_filter(u, v) = ∫(ε^2 * (∇(v) ⋅ ∇(u)) + v * u) * dΩ
l_filter(v) = ∫((ρh * v)) * dΩ
FILTER = StaticLinearModel(l_filter, a_filter, Vρ, Vρ, D_bc)

ρfh = get_state(FILTER)

# ************************************************************************************************
#                                 Forward Problem
# ************************************************************************************************
km = Kinematics(Mechano, Solid)
F, _, _ = get_Kinematics(km)
ΔDirac(ϵ, x₁, x₂) = (x) -> (x[1] ≥ x₁ - ϵ && x[1] ≤ x₁ + ϵ && x[2] ≥ x₂ - ϵ && x[2] ≤ x₂ + ϵ) ? 1.0 : 0.0
g(Λ) = (x) -> ΔDirac(0.1, xmax, ymax / 2)(x) * VectorValue(0.0, -Λ * 1e-3)
res(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ ∘ (ρfh, F ∘ (∇(u)'))))dΩ - ∫(g(Λ) ⋅ v)dΓ
jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂2Ψ ∘ (ρfh, F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩ
ls = LUSolver()
nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-10, rtol=1.e-8, verbose=false)
 
FORWARD = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls_)

uh = get_state(FORWARD)

# ************************************************************************************************
#                                 Adjoint Problem
# ************************************************************************************************

a(p, v) = jac(1.0)(uh, p, v)
l(v) = ∫(g(1.0) ⋅ v)dΓ
ls = LUSolver()
ADJOINT = StaticLinearModel(l, a, V, V, D_bc)

ph = get_state(ADJOINT)
# ************************************************************************************************
#                                 Optimisation functionals
# ************************************************************************************************

J_(u, ϕ) = ∫(g(1.0) ⋅ u)dΓ
DJ_(u, p, ϕ) = (v) -> (-1.0) * ∫(((D∂Ψ_Dρ ∘ (ϕ, F ∘ (∇(u)'))) ⊙ (∇(p)')) * v)dΩ
Compliance = FEFunctional(J_, DJ_, uh, ph, ρfh)

Vol_(u, ϕ) = ∫(ϕ)dΩ
DVol_(u, p, ϕ) = (v) -> ∫(v)dΩ
Volume = FEFunctional(Vol_, DVol_, nothing, nothing, ρh)

# ************************************************************************************************
#                                 Optimization loop
# ************************************************************************************************

tol = 1e-3
λ = 3000
t = 1
V0 = ∑(∫(1.0)dΩ)
pvd_topo = paraview_collection(simdir, append=false)

PDES = [FILTER, FORWARD, ADJOINT]

options = Dict(
  FILTER => NamedTuple(),
  FORWARD => (stepping=(nsteps=5, maxbisec=5),),
  ADJOINT => NamedTuple(),
)

ρv = get_free_dof_values(ρh)

  for i_top in 1:400

    # SOLVE PDEs
    map(x -> solve!(x; options[x]...), PDES)

    # Evaluate Functionals
    J, dJ = evaluate!(Compliance)
    Vol, dVol = evaluate!(Volume)

    if i_top == 1
      J, dJ = adimensionalize!(Compliance, J)
      Vol, dVol = adimensionalize!(Volume, V0)
      # options[FORWARD]=(stepping=(nsteps=1, maxbisec=5),ProjectDirichlet=false)
    end

    FILTER(dJ)

    DJh = FEFunction(Vρ, dJ)
    DJh_L2 = interpolate_everywhere(DJh, VρL2)
    DJL2 = get_free_dof_values(DJh_L2)

    Jt = J + λ * Vol
    DJt = DJL2 + λ * dVol
    ρv .-= t * DJt
    ρv .= map((x) -> max(0.0, min(x, 1.0)), ρv)
    @show J, Vol, Jt, norm(DJt), minimum(ρv), maximum(ρv)
    if norm(DJt) < tol
      break
    else
      fileI = simdir * "/iter_$i_top" * ".vtu"
      pvd_topo[i_top] = createvtk(Ω, fileI, cellfields=["uh" => uh, "ρh" => ρh, "DJh" => DJt])
    end
  end
  WriteVTK.vtk_save(pvd_topo)

