using Gridap, GridapGmsh, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using GridapSolvers.LinearSolvers
using HyperFEM

using Gridap.FESpaces
using WriteVTK

# using Gridap.MultiField
# using Gridap.FESpaces: get_assembly_strategy
# using Gridap.Algebra
# using BlockArrays

# using ForwardDiff
# using LinearAlgebra


pname = "Hyperelastic_Cylinder"
simdir = datadir("sims", pname)
setupfolder(simdir)

mesh_file = joinpath("./data/models/Cylinder_mesh.msh")
model = GmshDiscreteModel(mesh_file)

# Constitutive model
μParams = [37.2e3, 37.2e3]
physmodel_iso = MooneyRivlin3D(λ=(μParams[1] + μParams[2]) * 1e2, μ1=μParams[1], μ2=μParams[2])
c1 = [4.68e-3, 2.9e-2, 28.16e3, 28.16e3]
c2 = [2.97e-7, 1.68e-6, 3.48, 3.48]
physmodel_transiso = HGO_4Fibers(c1=c1, c2=c2)
physmodel = physmodel_iso + physmodel_transiso

# Functionals for Energy and Analytical derivatives
Ψ, ∂ΨF, ∂ΨFF = physmodel()

# Setup integration
Ω = Triangulation(model)
Γ = BoundaryTriangulation(model, tags="Internal")
order = 1   # order of interpolation
degree = 2 * order  # integration degree
dΩ = Measure(Ω, degree)
dΓ = Measure(Γ, degree)

# Dirichlet boundary conditions 
dir_u_tags = ["Fixed"]
dir_u_values = [[0.0, 0.0, 0.0]]
D_bc = DirichletBC(dir_u_tags, dir_u_values)

# Finite Elements
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffep = ReferenceFE(lagrangian, Float64, order)  # Finite Element for pressure
V = TestFESpace(model, reffeu, D_bc, conformity=:H1)
U = TrialFESpace(V, D_bc)
Vp = TestFESpace(Γ, reffep)  # Finite element space for pressure
Vfiber = FESpace(model, reffeu, conformity=:H1)

#  FE function for the normal vector in the internal surface
VNΓ = TestFESpace(Γ, reffeu, conformity=:H1)
NΓ = get_normal_vector(Γ)
Nh = interpolate_everywhere(NΓ, VNΓ)

# ---------------------
# Fibers configuration
# ---------------------
function fibres(x)
  n_ = [x[1], x[2], 0.0]
  c_ = [-x[2], x[1], 0.0]
  l_ = [0.0, 0.0, 1.0]
  n = VectorValue(n_) / norm(n_)
  c = VectorValue(c_) / norm(c_)
  l = VectorValue(l_)                
  return n, c, l
end

function RotatedVectors(α, l, c)
  n = l × c
  v1 = cos(α) * c + sin(α) * (n × c)
  v2 = cos(α + π / 2) * c + sin(α + π / 2) * (n × c)
  return v1, v2
end

ch = interpolate_everywhere(x -> fibres(x)[2], Vfiber)   # circumferential vector
lh = interpolate_everywhere(x -> fibres(x)[3], Vfiber)   # longitudinal vector
v1h = interpolate_everywhere(x -> RotatedVectors(π / 4, fibres(x)[3], fibres(x)[2])[1], Vfiber)   # vector between c and l at an angle α from c
v2h = interpolate_everywhere(x -> RotatedVectors(π / 4, fibres(x)[3], fibres(x)[2])[2], Vfiber)   # vector between c and l at an angle α+π/2 from c


# Kinematic Description
km=Kinematics(Mechano,Solid)
F, H, _ = get_Kinematics(km)   

p₀ = -50e2
ph = interpolate_everywhere(p₀, Vp)  # value of pressure in Internal surface

res(Λ) = (u, v) -> ∫((∂ΨF ∘ (F ∘ (∇(u)'), lh, ch, v1h, v2h)) ⊙ (∇(v)'))dΩ - Λ * ∫(ph * ((H ∘ (F ∘ (∇(u)'))) * Nh ⋅ v))dΓ
jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂ΨFF ∘ (F ∘ (∇(u)'), lh, ch, v1h, v2h)) ⊙ (∇(du)')))dΩ - Λ * ∫(ph * (((F ∘ (∇(u)') × ∇(du)') * Nh) ⋅ v))dΓ

# nonlinear solver
ls = LUSolver()  #Direct Solver
nls_ = Newton_RaphsonSolver(ls; maxiter=15, atol=1.e-10, rtol=1.e-7, verbose=true)

# Computational model
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls_)

# Postprocessor to save results
function driverpost(post; Ω=Ω, U=U)
  state = post.comp_model.caches[3]
  Λ_ = post.iter
  Λ = post.Λ[Λ_]
  xh = FEFunction(U, state)
  writevtk(Ω, simdir * "/Λ_$Λ_" * ".vtu",
    cellfields=["u" => xh, "ch" => ch, "lh" => lh, "v1" => v1h, "v2" => v2h, "Λ" => Λ])
end

post_model = PostProcessor(comp_model, driverpost; is_vtk=true, filepath=simdir)

x = solve!(comp_model; stepping=(nsteps=20, maxbisec=5), post=post_model)